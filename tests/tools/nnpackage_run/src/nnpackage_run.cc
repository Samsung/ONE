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
#include "randomgen.h"
#ifdef RUY_PROFILER
#include "ruy/profiler/profiler.h"
#endif

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <libgen.h>
#include <stdexcept>
#include <unordered_map>
#include <vector>

static const char *default_backend_cand = "cpu";

void overwriteShapeMap(nnpkg_run::TensorShapeMap &shape_map,
                       std::vector<nnpkg_run::TensorShape> shapes)
{
  for (uint32_t i = 0; i < shapes.size(); i++)
    shape_map[i] = shapes[i];
}

int main(const int argc, char **argv)
{
  using namespace nnpkg_run;

  try
  {
    Args args(argc, argv);
    auto nnpackage_path = args.getPackageFilename();
    if (args.printVersion())
    {
      uint32_t version;
      NNPR_ENSURE_STATUS(nnfw_query_info_u32(NULL, NNFW_INFO_ID_VERSION, &version));
      std::cout << "nnpkg_run (nnfw runtime: v" << (version >> 24) << "."
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
      NNPR_ENSURE_STATUS(nnfw_load_model_from_file(session, nnpackage_path.c_str()));
    });

    char *available_backends = std::getenv("BACKENDS");
    if (available_backends)
      NNPR_ENSURE_STATUS(nnfw_set_available_backends(session, available_backends));

    uint32_t num_inputs;
    NNPR_ENSURE_STATUS(nnfw_input_size(session, &num_inputs));

    // verify input and output

    auto verifyInputTypes = [session]() {
      uint32_t sz;
      NNPR_ENSURE_STATUS(nnfw_input_size(session, &sz));
      for (uint32_t i = 0; i < sz; ++i)
      {
        nnfw_tensorinfo ti;
        NNPR_ENSURE_STATUS(nnfw_input_tensorinfo(session, i, &ti));

        if (ti.dtype < NNFW_TYPE_TENSOR_FLOAT32 || ti.dtype > NNFW_TYPE_TENSOR_QUANT8_ASYMM_SIGNED)
        {
          std::cerr << "E: not supported input type" << std::endl;
          exit(-1);
        }
      }
    };

    auto verifyOutputTypes = [session]() {
      uint32_t sz;
      NNPR_ENSURE_STATUS(nnfw_output_size(session, &sz));

      for (uint32_t i = 0; i < sz; ++i)
      {
        nnfw_tensorinfo ti;
        NNPR_ENSURE_STATUS(nnfw_output_tensorinfo(session, i, &ti));

        if (ti.dtype < NNFW_TYPE_TENSOR_FLOAT32 || ti.dtype > NNFW_TYPE_TENSOR_QUANT8_ASYMM_SIGNED)
        {
          std::cerr << "E: not supported output type" << std::endl;
          exit(-1);
        }
      }
    };

    auto setTensorInfo = [session](const TensorShapeMap &tensor_shape_map) {
      for (auto tensor_shape : tensor_shape_map)
      {
        auto ind = tensor_shape.first;
        auto &shape = tensor_shape.second;
        nnfw_tensorinfo ti;
        // to fill dtype
        NNPR_ENSURE_STATUS(nnfw_input_tensorinfo(session, ind, &ti));

        bool set_input = false;
        if (ti.rank != shape.size())
        {
          set_input = true;
        }
        else
        {
          for (int i = 0; i < ti.rank; i++)
          {
            if (ti.dims[i] != shape.at(i))
            {
              set_input = true;
              break;
            }
          }
        }
        if (!set_input)
          continue;

        ti.rank = shape.size();
        for (int i = 0; i < ti.rank; i++)
          ti.dims[i] = shape.at(i);
        NNPR_ENSURE_STATUS(nnfw_set_input_tensorinfo(session, ind, &ti));
      }
    };

    verifyInputTypes();
    verifyOutputTypes();

// set input shape before compilation
#if defined(ONERT_HAVE_HDF5) && ONERT_HAVE_HDF5 == 1

    auto fill_shape_from_h5 = [&session](const std::string &h5_file, TensorShapeMap &shape_map) {
      assert(!h5_file.empty());
      auto shapes = H5Formatter(session).readTensorShapes(h5_file);
      overwriteShapeMap(shape_map, shapes);
    };

    if (args.getWhenToUseH5Shape() == WhenToUseH5Shape::PREPARE)
      fill_shape_from_h5(args.getLoadFilename(), args.getShapeMapForPrepare());
#endif
    setTensorInfo(args.getShapeMapForPrepare());

    // prepare execution

    // TODO When nnfw_{prepare|run} are failed, can't catch the time
    phases.run("PREPARE", [&](const benchmark::Phase &, uint32_t) {
      NNPR_ENSURE_STATUS(nnfw_prepare(session));
    });

// set input shape after compilation and before execution
#if defined(ONERT_HAVE_HDF5) && ONERT_HAVE_HDF5 == 1
    if (args.getWhenToUseH5Shape() == WhenToUseH5Shape::RUN ||
        (!args.getLoadFilename().empty() && !args.shapeParamProvided()))
      fill_shape_from_h5(args.getLoadFilename(), args.getShapeMapForRun());
#endif
    setTensorInfo(args.getShapeMapForRun());

    // prepare input
    std::vector<Allocation> inputs(num_inputs);
#if defined(ONERT_HAVE_HDF5) && ONERT_HAVE_HDF5 == 1
    if (!args.getLoadFilename().empty())
      H5Formatter(session).loadInputs(args.getLoadFilename(), inputs);
    else
      RandomGenerator(session).generate(inputs);
#else
    RandomGenerator(session).generate(inputs);
#endif

    // prepare output
    uint32_t num_outputs = 0;
    NNPR_ENSURE_STATUS(nnfw_output_size(session, &num_outputs));
    std::vector<Allocation> outputs(num_outputs);
    auto output_sizes = args.getOutputSizes();
    for (uint32_t i = 0; i < num_outputs; i++)
    {
      nnfw_tensorinfo ti;
      uint64_t output_size_in_bytes = 0;
      {
        auto found = output_sizes.find(i);
        if (found == output_sizes.end())
        {
          NNPR_ENSURE_STATUS(nnfw_output_tensorinfo(session, i, &ti));
          output_size_in_bytes = bufsize_for(&ti);
        }
        else
        {
          output_size_in_bytes = found->second;
        }
      }
      outputs[i].alloc(output_size_in_bytes);
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
        [&](const benchmark::Phase &, uint32_t) { for (int i = 0; i < 1000; ++i) NNPR_ENSURE_STATUS(nnfw_run(session)); },
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
        },
        args.getNumRuns(), true);
    }

#if defined(ONERT_HAVE_HDF5) && ONERT_HAVE_HDF5 == 1
    // dump output tensors
    if (!args.getDumpFilename().empty())
      H5Formatter(session).dumpOutputs(args.getDumpFilename(), outputs);
#endif

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
      char *res = realpath(nnpackage_path.c_str(), buf);
      if (res)
      {
        nnpkg_basename = basename(buf);
      }
      else
      {
        std::cerr << "E: during getting realpath from nnpackage_path." << std::endl;
        exit(-1);
      }
      exec_basename = basename(argv[0]);
    }

    benchmark::writeResult(result, exec_basename, nnpkg_basename, backend_name);

    return 0;
  }
  catch (std::runtime_error &e)
  {
    std::cerr << "E: Fail to run by runtime error:" << e.what() << std::endl;
    exit(-1);
  }
}
