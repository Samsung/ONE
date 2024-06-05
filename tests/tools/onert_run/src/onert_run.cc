/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "allocation.h"
#include "args.h"
#include "benchmark.h"
#if defined(ONERT_HAVE_HDF5) && ONERT_HAVE_HDF5 == 1
#include "h5formatter.h"
#endif
#include "nnfw.h"
#include "nnfw_util.h"
#include "nnfw_internal.h"
#include "nnfw_experimental.h"
#include "randomgen.h"
#include "rawformatter.h"
#ifdef RUY_PROFILER
#include "ruy/profiler/profiler.h"
#endif

#include <boost/program_options.hpp>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <libgen.h>
#include <stdexcept>
#include <unordered_map>
#include <vector>

static const char *default_backend_cand = "cpu";

void overwriteShapeMap(onert_run::TensorShapeMap &shape_map,
                       std::vector<onert_run::TensorShape> shapes)
{
  for (uint32_t i = 0; i < shapes.size(); i++)
    shape_map[i] = shapes[i];
}

std::string genQuantizedModelPathFromModelPath(const std::string &model_path,
                                               NNFW_QUANTIZE_TYPE qtype)
{
  auto const extension_pos = model_path.find(".circle");
  if (extension_pos == std::string::npos)
  {
    std::cerr << "Input model isn't .circle." << std::endl;
    exit(-1);
  }
  switch (qtype)
  {
    case NNFW_QUANTIZE_TYPE_U8_ASYM:
      return model_path.substr(0, extension_pos) + "_quantized_q8.circle";
    case NNFW_QUANTIZE_TYPE_I16_SYM:
      return model_path.substr(0, extension_pos) + "_quantized_q16.circle";
    case NNFW_QUANTIZE_TYPE_WO_I8_SYM:
      return model_path.substr(0, extension_pos) + "_quantized_q8wo.circle";
    case NNFW_QUANTIZE_TYPE_WO_I16_SYM:
      return model_path.substr(0, extension_pos) + "_quantized_q16wo.circle";
  }

  throw std::runtime_error{"Invalid quantization type"};
}

std::string genQuantizedModelPathFromPackagePath(const std::string &package_path,
                                                 NNFW_QUANTIZE_TYPE qtype)
{
  auto package_path_without_slash = package_path;
  if (package_path_without_slash.back() == '/')
    package_path_without_slash.pop_back();
  auto package_name_pos = package_path_without_slash.find_last_of('/');
  if (package_name_pos == std::string::npos)
    package_name_pos = 0;
  else
    package_name_pos++;
  auto package_name = package_path_without_slash.substr(package_name_pos);
  switch (qtype)
  {
    case NNFW_QUANTIZE_TYPE_U8_ASYM:
      return package_path_without_slash + "/" + package_name + "_quantized_q8.circle";
    case NNFW_QUANTIZE_TYPE_I16_SYM:
      return package_path_without_slash + "/" + package_name + "_quantized_q16.circle";
    case NNFW_QUANTIZE_TYPE_WO_I8_SYM:
      return package_path_without_slash + "/" + package_name + "_quantized_q8wo.circle";
    case NNFW_QUANTIZE_TYPE_WO_I16_SYM:
      return package_path_without_slash + "/" + package_name + "_quantized_q16wo.circle";
  }

  throw std::runtime_error{"Invalid quantization type"};
}

int main(const int argc, char **argv)
{
  using namespace onert_run;

  try
  {
    Args args(argc, argv);
    if (args.printVersion())
    {
      uint32_t version;
      NNPR_ENSURE_STATUS(nnfw_query_info_u32(NULL, NNFW_INFO_ID_VERSION, &version));
      std::cout << "onert_run (nnfw runtime: v" << (version >> 24) << "."
                << ((version & 0x0000FF00) >> 8) << "." << (version & 0xFF) << ")" << std::endl;
      exit(0);
    }

#ifdef RUY_PROFILER
    ruy::profiler::ScopeProfile ruy_profile;
#endif

    // TODO Apply verbose level to phases
    const int verbose = args.getVerboseLevel();
    benchmark::Phases phases(
      benchmark::PhaseOption{args.getMemoryPoll(), args.getGpuMemoryPoll(), args.getRunDelay()});

    nnfw_session *session = nullptr;
    NNPR_ENSURE_STATUS(nnfw_create_session(&session));

    // ModelLoad
    phases.run("MODEL_LOAD", [&](const benchmark::Phase &, uint32_t) {
      if (args.useSingleModel())
        NNPR_ENSURE_STATUS(
          nnfw_load_model_from_modelfile(session, args.getModelFilename().c_str()));
      else
        NNPR_ENSURE_STATUS(nnfw_load_model_from_file(session, args.getPackageFilename().c_str()));
    });

    // Quantize model
    auto quantize = args.getQuantize();
    if (!quantize.empty())
    {
      NNFW_QUANTIZE_TYPE quantize_type = NNFW_QUANTIZE_TYPE_NOT_SET;
      if (quantize == "uint8")
        quantize_type = NNFW_QUANTIZE_TYPE_U8_ASYM;
      if (quantize == "int16")
        quantize_type = NNFW_QUANTIZE_TYPE_I16_SYM;
      if (quantize == "int8_wo")
        quantize_type = NNFW_QUANTIZE_TYPE_WO_I8_SYM;
      if (quantize == "int16_wo")
        quantize_type = NNFW_QUANTIZE_TYPE_WO_I16_SYM;
      NNPR_ENSURE_STATUS(nnfw_set_quantization_type(session, quantize_type));

      if (args.getQuantizedModelPath() != "")
        NNPR_ENSURE_STATUS(
          nnfw_set_quantized_model_path(session, args.getQuantizedModelPath().c_str()));
      else
      {
        if (args.useSingleModel())
          NNPR_ENSURE_STATUS(nnfw_set_quantized_model_path(
            session,
            genQuantizedModelPathFromModelPath(args.getModelFilename(), quantize_type).c_str()));
        else
          NNPR_ENSURE_STATUS(nnfw_set_quantized_model_path(
            session, genQuantizedModelPathFromPackagePath(args.getPackageFilename(), quantize_type)
                       .c_str()));
      }

      NNPR_ENSURE_STATUS(nnfw_quantize(session));
    }

    // Generate target backend code
    auto codegen = args.getCodegen();
    if (!codegen.empty())
    {
      NNPR_ENSURE_STATUS(nnfw_set_codegen_model_path(session, args.getCodegenModelPath().c_str()));
      NNPR_ENSURE_STATUS(nnfw_codegen(session, codegen.c_str(), NNFW_CODEGEN_PREF_DEFAULT));
    }

    char *available_backends = std::getenv("BACKENDS");
    if (available_backends)
      NNPR_ENSURE_STATUS(nnfw_set_available_backends(session, available_backends));

    uint32_t num_inputs;
    uint32_t num_outputs;
    NNPR_ENSURE_STATUS(nnfw_input_size(session, &num_inputs));
    NNPR_ENSURE_STATUS(nnfw_output_size(session, &num_outputs));

    // verify input and output
    for (uint32_t i = 0; i < num_inputs; ++i)
    {
      nnfw_tensorinfo ti;
      NNPR_ENSURE_STATUS(nnfw_input_tensorinfo(session, i, &ti));

      if (ti.dtype < NNFW_TYPE_TENSOR_FLOAT32 || ti.dtype > NNFW_TYPE_TENSOR_QUANT16_SYMM_SIGNED)
      {
        std::cerr << "E: not supported input type" << std::endl;
        exit(-1);
      }
    }

    for (uint32_t i = 0; i < num_outputs; ++i)
    {
      nnfw_tensorinfo ti;
      NNPR_ENSURE_STATUS(nnfw_output_tensorinfo(session, i, &ti));

      if (ti.dtype < NNFW_TYPE_TENSOR_FLOAT32 || ti.dtype > NNFW_TYPE_TENSOR_QUANT16_SYMM_SIGNED)
      {
        std::cerr << "E: not supported output type" << std::endl;
        exit(-1);
      }
    }

    std::vector<Allocation> inputs(num_inputs);
    std::vector<Allocation> outputs(num_outputs);

    auto setInputTensorInfo = [&](const TensorShapeMap &tensor_shape_map, bool allocate) {
      for (uint32_t i = 0; i < num_inputs; i++)
      {
        nnfw_tensorinfo ti;
        NNPR_ENSURE_STATUS(nnfw_input_tensorinfo(session, i, &ti));

        // Find updated shape index and update tensor info
        auto found = tensor_shape_map.find(i);
        if (found != tensor_shape_map.end())
        {
          auto &shape = found->second;
          bool set_input = false;
          if (ti.rank != shape.size())
          {
            set_input = true;
          }
          else
          {
            for (int32_t i = 0; i < ti.rank; i++)
            {
              if (ti.dims[i] != shape.at(i))
              {
                set_input = true;
                break;
              }
            }
          }

          if (set_input)
          {
            ti.rank = shape.size();
            for (int i = 0; i < ti.rank; i++)
              ti.dims[i] = shape.at(i);
            NNPR_ENSURE_STATUS(nnfw_set_input_tensorinfo(session, i, &ti));
          }
        }

        // Allocate memory for input data and set buffer
        if (allocate)
        {
          auto input_size_in_bytes = bufsize_for(&ti);
          inputs[i].alloc(input_size_in_bytes, ti.dtype);

          NNPR_ENSURE_STATUS(
            nnfw_set_input(session, i, ti.dtype, inputs[i].data(), input_size_in_bytes));
          NNPR_ENSURE_STATUS(nnfw_set_input_layout(session, i, NNFW_LAYOUT_CHANNELS_LAST));
        }
      }
    };

// set input shape before compilation
#if defined(ONERT_HAVE_HDF5) && ONERT_HAVE_HDF5 == 1

    auto fill_shape_from_h5 = [&](const std::string &h5_file, TensorShapeMap &shape_map) {
      assert(!h5_file.empty());
      auto shapes = H5Formatter().readTensorShapes(h5_file, num_inputs);
      overwriteShapeMap(shape_map, shapes);
    };

    if (args.getWhenToUseH5Shape() == WhenToUseH5Shape::PREPARE)
      fill_shape_from_h5(args.getLoadFilename(), args.getShapeMapForPrepare());
#endif
    // Set shape info, but don't alloc yet
    setInputTensorInfo(args.getShapeMapForPrepare(), false);

    // prepare execution

    // TODO When nnfw_{prepare|run} are failed, can't catch the time
    phases.run("PREPARE", [&](const benchmark::Phase &, uint32_t) {
      NNPR_ENSURE_STATUS(nnfw_prepare(session));
    });

    // Set input shape and buffer after compilation and before execution
#if defined(ONERT_HAVE_HDF5) && ONERT_HAVE_HDF5 == 1
    if (args.getWhenToUseH5Shape() == WhenToUseH5Shape::RUN ||
        (!args.getLoadFilename().empty() && !args.shapeParamProvided()))
      fill_shape_from_h5(args.getLoadFilename(), args.getShapeMapForRun());
#endif
    setInputTensorInfo(args.getShapeMapForRun(), true);

    // Prepare input data
    auto random_generator = RandomGenerator();
    bool regenerate_input = false;
    if (!args.getLoadRawFilename().empty())
      RawFormatter().loadInputs(args.getLoadRawFilename(), inputs);
#if defined(ONERT_HAVE_HDF5) && ONERT_HAVE_HDF5 == 1
    else if (!args.getLoadFilename().empty())
      H5Formatter().loadInputs(args.getLoadFilename(), inputs);
#endif
    else
    {
      random_generator.generate(inputs);
      // Set regenerate_input to true if input is random data and num_runs > 1
      // Ignore random generator is not used
      if (args.getNumRuns() > 1 && !args.getFixedInput())
        regenerate_input = true;
    }

    // Prepare output buffer
    auto output_sizes = args.getOutputSizes();
    for (uint32_t i = 0; i < num_outputs; i++)
    {
      nnfw_tensorinfo ti;
      NNPR_ENSURE_STATUS(nnfw_output_tensorinfo(session, i, &ti));
      uint64_t output_size_in_bytes = bufsize_for(&ti);
      {
        auto found = output_sizes.find(i);
        if (output_sizes.find(i) != output_sizes.end())
        {
          output_size_in_bytes = found->second;
        }
      }
      outputs[i].alloc(output_size_in_bytes, ti.dtype);
      NNPR_ENSURE_STATUS(
        nnfw_set_output(session, i, ti.dtype, outputs[i].data(), output_size_in_bytes));
      NNPR_ENSURE_STATUS(nnfw_set_output_layout(session, i, NNFW_LAYOUT_CHANNELS_LAST));
    }

    // NOTE: Measuring memory can't avoid taking overhead. Therefore, memory will be measured on the
    // only warmup.
    if (verbose == 0)
    {
      phases.run(
        "WARMUP",
        [&](const benchmark::Phase &, uint32_t) { NNPR_ENSURE_STATUS(nnfw_run(session)); },
        args.getWarmupRuns());
      phases.run(
        "EXECUTE",
        [&](const benchmark::Phase &, uint32_t) { NNPR_ENSURE_STATUS(nnfw_run(session)); },
        [&](const benchmark::Phase &, uint32_t) {
          if (regenerate_input)
            random_generator.generate(inputs);
        },
        args.getNumRuns(), true);
    }
    else
    {
      phases.run(
        "WARMUP",
        [&](const benchmark::Phase &, uint32_t) { NNPR_ENSURE_STATUS(nnfw_run(session)); },
        [&](const benchmark::Phase &phase, uint32_t nth) {
          std::cout << "... "
                    << "warmup " << nth + 1 << " takes " << phase.time[nth] / 1e3 << " ms"
                    << std::endl;
        },
        args.getWarmupRuns());
      phases.run(
        "EXECUTE",
        [&](const benchmark::Phase &, uint32_t) { NNPR_ENSURE_STATUS(nnfw_run(session)); },
        [&](const benchmark::Phase &phase, uint32_t nth) {
          std::cout << "... "
                    << "run " << nth + 1 << " takes " << phase.time[nth] / 1e3 << " ms"
                    << std::endl;
          if (regenerate_input)
            random_generator.generate(inputs);
        },
        args.getNumRuns(), true);
    }

    // Check dump conditions
    // Do not dump if not fixed random input
    if (regenerate_input)
    {
      bool cannot_dump = false;
#if defined(ONERT_HAVE_HDF5) && ONERT_HAVE_HDF5 == 1
      if (!args.getDumpFilename().empty())
        cannot_dump = true;
#endif
      if (!args.getDumpRawInputFilename().empty() || !args.getDumpRawFilename().empty())
        cannot_dump = true;
      if (cannot_dump)
        throw std::runtime_error("Cannot dump input/output with inputs regeneration");
    }

#if defined(ONERT_HAVE_HDF5) && ONERT_HAVE_HDF5 == 1
    // dump output tensors
    if (!args.getDumpFilename().empty())
    {
      std::vector<TensorShape> output_shapes;
      for (uint32_t i = 0; i < num_outputs; i++)
      {
        nnfw_tensorinfo ti;
        NNPR_ENSURE_STATUS(nnfw_output_tensorinfo(session, i, &ti));

        TensorShape shape;
        for (uint32_t j = 0; j < ti.rank; j++)
          shape.emplace_back(ti.dims[j]);

        output_shapes.emplace_back(shape);
      }

      H5Formatter().dumpOutputs(args.getDumpFilename(), outputs, output_shapes);
    }
#endif
    if (!args.getDumpRawInputFilename().empty())
      RawFormatter().dumpInputs(args.getDumpRawInputFilename(), inputs);
    if (!args.getDumpRawFilename().empty())
      RawFormatter().dumpOutputs(args.getDumpRawFilename(), outputs);

    NNPR_ENSURE_STATUS(nnfw_close_session(session));

    // TODO Apply verbose level to result

    // prepare result
    benchmark::Result result(phases);

    // to stdout
    benchmark::printResult(result);

    // to csv
    if (args.getWriteReport() == false)
      return 0;

    // prepare csv task
    std::string exec_basename;
    std::string nnpkg_basename;
    std::string backend_name = (available_backends) ? available_backends : default_backend_cand;
    {
      char buf[PATH_MAX];
      char *res = args.useSingleModel() ? realpath(args.getModelFilename().c_str(), buf)
                                        : realpath(args.getPackageFilename().c_str(), buf);
      if (res)
      {
        nnpkg_basename = basename(buf);
      }
      else
      {
        std::cerr << "E: during getting realpath from nnpackage or model path." << std::endl;
        exit(-1);
      }
      exec_basename = basename(argv[0]);
    }

    benchmark::writeResult(result, exec_basename, nnpkg_basename, backend_name);

    return 0;
  }
  catch (boost::program_options::error &e)
  {
    std::cerr << "E: " << e.what() << std::endl;
    exit(-1);
  }
  catch (std::runtime_error &e)
  {
    std::cerr << "E: Fail to run by runtime error:" << e.what() << std::endl;
    exit(-1);
  }
}
