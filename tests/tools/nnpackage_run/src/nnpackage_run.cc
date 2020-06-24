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
#include "tflite/Diff.h"
#include "nnfw.h"
#include "nnfw_util.h"
#include "nnfw_debug.h"
#ifdef RUY_PROFILER
#include "ruy/profiler/profiler.h"
#endif

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <libgen.h>

namespace nnpkg_run
{

template <class T> void randomData(nnfw::misc::RandomGenerator &randgen, void *data, uint64_t size)
{
  for (uint64_t i = 0; i < size; i++)
    reinterpret_cast<T *>(data)[i] = randgen.generate<T>();
}
}

static const char *default_backend_cand = "acl_cl";

NNFW_STATUS resolve_op_backend(nnfw_session *session)
{
  static std::unordered_map<std::string, std::string> operation_map = {
      {"TRANSPOSE_CONV", "OP_BACKEND_TransposeConv"},      {"CONV_2D", "OP_BACKEND_Conv2D"},
      {"DEPTHWISE_CONV_2D", "OP_BACKEND_DepthwiseConv2D"}, {"MEAN", "OP_BACKEND_Mean"},
      {"AVERAGE_POOL_2D", "OP_BACKEND_AvgPool2D"},         {"MAX_POOL_2D", "OP_BACKEND_MaxPool2D"},
      {"INSTANCE_NORM", "OP_BACKEND_InstanceNorm"},        {"ADD", "OP_BACKEND_Add"}};

  for (auto i : operation_map)
  {
    char *default_backend = std::getenv(i.second.c_str());
    if (default_backend)
    {
      NNFW_STATUS return_result = nnfw_set_op_backend(session, i.first.c_str(), default_backend);
      if (return_result == NNFW_STATUS_ERROR)
        return return_result;
    }
  }

  return NNFW_STATUS_NO_ERROR;
}

int main(const int argc, char **argv)
{
  using namespace nnpkg_run;
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

  try
  {

#ifdef RUY_PROFILER
    ruy::profiler::ScopeProfile ruy_profile;
#endif

    benchmark::Phases phases(benchmark::PhaseOption{args.getMemoryPoll(), args.getGpuMemoryPoll()});

    nnfw_session *session = nullptr;
    NNPR_ENSURE_STATUS(nnfw_create_session(&session));

    // ModelLoad
    phases.run("MODEL_LOAD", [&](const benchmark::Phase &, uint32_t) {
      NNPR_ENSURE_STATUS(nnfw_load_model_from_file(session, nnpackage_path.c_str()));
    });

    char *available_backends = std::getenv("BACKENDS");
    if (available_backends)
      NNPR_ENSURE_STATUS(nnfw_set_available_backends(session, available_backends));
    NNPR_ENSURE_STATUS(resolve_op_backend(session));

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

        if (ti.dtype < NNFW_TYPE_TENSOR_FLOAT32 || ti.dtype > NNFW_TYPE_TENSOR_INT64)
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

        if (ti.dtype < NNFW_TYPE_TENSOR_FLOAT32 || ti.dtype > NNFW_TYPE_TENSOR_INT64)
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

        ti.rank = shape.size();
        for (int i = 0; i < ti.rank; i++)
          ti.dims[i] = shape.at(i);
        NNPR_ENSURE_STATUS(nnfw_set_input_tensorinfo(session, ind, &ti));
      }
    };

    verifyInputTypes();
    verifyOutputTypes();

    // set input shape before compilation
    setTensorInfo(args.getShapeMapForPrepare());

    // prepare execution

    // TODO When nnfw_{prepare|run} are failed, can't catch the time
    phases.run("PREPARE", [&](const benchmark::Phase &, uint32_t) {
      NNPR_ENSURE_STATUS(nnfw_prepare(session));
    });

    // set input shape after compilation and before execution
    setTensorInfo(args.getShapeMapForRun());

    // prepare input
    std::vector<Allocation> inputs(num_inputs);

    auto generateInputs = [session, num_inputs, &inputs]() {
      // generate random data
      const int seed = 1;
      nnfw::misc::RandomGenerator randgen{seed, 0.0f, 2.0f};
      for (uint32_t i = 0; i < num_inputs; ++i)
      {
        nnfw_tensorinfo ti;
        NNPR_ENSURE_STATUS(nnfw_input_tensorinfo(session, i, &ti));
        auto input_size_in_bytes = bufsize_for(&ti);
        inputs[i].alloc(input_size_in_bytes);
        switch (ti.dtype)
        {
          case NNFW_TYPE_TENSOR_FLOAT32:
            randomData<float>(randgen, inputs[i].data(), num_elems(&ti));
            break;
          case NNFW_TYPE_TENSOR_QUANT8_ASYMM:
            randomData<uint8_t>(randgen, inputs[i].data(), num_elems(&ti));
            break;
          case NNFW_TYPE_TENSOR_BOOL:
            randomData<bool>(randgen, inputs[i].data(), num_elems(&ti));
            break;
          case NNFW_TYPE_TENSOR_UINT8:
            randomData<uint8_t>(randgen, inputs[i].data(), num_elems(&ti));
            break;
          case NNFW_TYPE_TENSOR_INT32:
            randomData<int32_t>(randgen, inputs[i].data(), num_elems(&ti));
            break;
          case NNFW_TYPE_TENSOR_INT64:
            randomData<int64_t>(randgen, inputs[i].data(), num_elems(&ti));
            break;
          default:
            std::cerr << "Not supported input type" << std::endl;
            std::exit(-1);
        }
        NNPR_ENSURE_STATUS(
            nnfw_set_input(session, i, ti.dtype, inputs[i].data(), input_size_in_bytes));
        NNPR_ENSURE_STATUS(nnfw_set_input_layout(session, i, NNFW_LAYOUT_CHANNELS_LAST));
      }
    };
#if defined(ONERT_HAVE_HDF5) && ONERT_HAVE_HDF5 == 1
    if (!args.getLoadFilename().empty())
      H5Formatter(session).loadInputs(args.getLoadFilename(), inputs);
    else
      generateInputs();
#else
    generateInputs();
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
    // warmup runs
    phases.run("WARMUP",
               [&](const benchmark::Phase &, uint32_t) { NNPR_ENSURE_STATUS(nnfw_run(session)); },
               [&](const benchmark::Phase &phase, uint32_t nth) {
                 std::cout << "... "
                           << "warmup " << nth + 1 << " takes " << phase.time[nth] / 1e3 << " ms"
                           << std::endl;
               },
               args.getWarmupRuns());

    // actual runs
    phases.run("EXECUTE",
               [&](const benchmark::Phase &, uint32_t) { NNPR_ENSURE_STATUS(nnfw_run(session)); },
               [&](const benchmark::Phase &phase, uint32_t nth) {
                 std::cout << "... "
                           << "run " << nth + 1 << " takes " << phase.time[nth] / 1e3 << " ms"
                           << std::endl;
               },
               args.getNumRuns(), true);

#if defined(ONERT_HAVE_HDF5) && ONERT_HAVE_HDF5 == 1
    // dump output tensors
    if (!args.getDumpFilename().empty())
      H5Formatter(session).dumpOutputs(args.getDumpFilename(), outputs);
#endif

    NNPR_ENSURE_STATUS(nnfw_close_session(session));

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
