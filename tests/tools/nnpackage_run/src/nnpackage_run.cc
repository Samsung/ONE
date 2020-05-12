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
#include "h5formatter.h"
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

namespace nnpkg_run
{

template <class T> void randomData(RandomGenerator &randgen, void *data, uint64_t size)
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

#ifdef RUY_PROFILER
  ruy::profiler::ScopeProfile ruy_profile;
#endif

  std::unique_ptr<benchmark::MemoryPoller> mp{nullptr};
  if (args.getMemoryPoll())
  {
    try
    {
      mp.reset(new benchmark::MemoryPoller(std::chrono::milliseconds(5), args.getGpuMemoryPoll()));
    }
    catch (const std::runtime_error &error)
    {
      std::cerr << error.what() << std::endl;
      return 1;
    }
  }

  nnfw_session *session = nullptr;
  NNPR_ENSURE_STATUS(nnfw_create_debug_session(&session));
  char *available_backends = std::getenv("BACKENDS");
  if (available_backends)
    NNPR_ENSURE_STATUS(nnfw_set_available_backends(session, available_backends));
  NNPR_ENSURE_STATUS(resolve_op_backend(session));

  // ModelLoad
  if (mp)
    mp->start(benchmark::Phase::MODEL_LOAD);
  uint64_t t_model_load = benchmark::nowMicros();
  NNPR_ENSURE_STATUS(nnfw_load_model_from_file(session, nnpackage_path.c_str()));
  t_model_load = benchmark::nowMicros() - t_model_load;
  if (mp)
    mp->end(benchmark::Phase::MODEL_LOAD);

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
      if (ti.dtype < NNFW_TYPE_TENSOR_FLOAT32 || ti.dtype > NNFW_TYPE_TENSOR_UINT8)
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
      if (ti.dtype < NNFW_TYPE_TENSOR_FLOAT32 || ti.dtype > NNFW_TYPE_TENSOR_UINT8)
      {
        std::cerr << "E: not supported output type" << std::endl;
        exit(-1);
      }
    }
  };

  verifyInputTypes();
  verifyOutputTypes();

  // prepare execution

  // TODO When nnfw_{prepare|run} are failed, can't catch the time
  if (mp)
    mp->start(benchmark::Phase::PREPARE);
  uint64_t t_prepare = benchmark::nowMicros();
  NNPR_ENSURE_STATUS(nnfw_prepare(session));
  t_prepare = benchmark::nowMicros() - t_prepare;
  if (mp)
    mp->end(benchmark::Phase::PREPARE);

  // prepare input
  std::vector<Allocation> inputs(num_inputs);

  auto generateInputs = [session, num_inputs, &inputs]() {
    // generate random data
    const int seed = 1;
    RandomGenerator randgen{seed, 0.0f, 2.0f};
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
        default:
          std::cerr << "Not supported input type" << std::endl;
          std::exit(-1);
      }
      NNPR_ENSURE_STATUS(
          nnfw_set_input(session, i, ti.dtype, inputs[i].data(), input_size_in_bytes));
      NNPR_ENSURE_STATUS(nnfw_set_input_layout(session, i, NNFW_LAYOUT_CHANNELS_LAST));
    }
  };
  if (!args.getLoadFilename().empty())
    H5Formatter(session).loadInputs(args.getLoadFilename(), inputs);
  else
    generateInputs();

  // prepare output

  uint32_t num_outputs = 0;
  NNPR_ENSURE_STATUS(nnfw_output_size(session, &num_outputs));
  std::vector<Allocation> outputs(num_outputs);

  for (uint32_t i = 0; i < num_outputs; i++)
  {
    nnfw_tensorinfo ti;
    NNPR_ENSURE_STATUS(nnfw_output_tensorinfo(session, i, &ti));
    auto output_size_in_bytes = bufsize_for(&ti);
    outputs[i].alloc(output_size_in_bytes);
    NNPR_ENSURE_STATUS(
        nnfw_set_output(session, i, ti.dtype, outputs[i].data(), output_size_in_bytes));
    NNPR_ENSURE_STATUS(nnfw_set_output_layout(session, i, NNFW_LAYOUT_CHANNELS_LAST));
  }

  // poll memories before warming up
  if (mp)
    mp->start(benchmark::Phase::EXECUTE);
  uint64_t run_us = benchmark::nowMicros();
  NNPR_ENSURE_STATUS(nnfw_run(session));
  run_us = benchmark::nowMicros() - run_us;
  if (mp)
    mp->end(benchmark::Phase::EXECUTE);

  // warmup runs
  for (uint32_t i = 1; i < args.getWarmupRuns(); i++)
  {
    uint64_t run_us = benchmark::nowMicros();
    NNPR_ENSURE_STATUS(nnfw_run(session));
    run_us = benchmark::nowMicros() - run_us;
    std::cout << "... "
              << "warmup " << i << " takes " << run_us / 1e3 << " ms" << std::endl;
  }

  // actual runs
  std::vector<double> t_execute;
  for (uint32_t i = 0; i < args.getNumRuns(); i++)
  {
    uint64_t run_us = benchmark::nowMicros();
    NNPR_ENSURE_STATUS(nnfw_run(session));
    run_us = benchmark::nowMicros() - run_us;
    t_execute.emplace_back(run_us);
    std::cout << "... "
              << "run " << i << " takes " << run_us / 1e3 << " ms" << std::endl;
  }

  // dump output tensors
  if (!args.getDumpFilename().empty())
    H5Formatter(session).dumpOutputs(args.getDumpFilename(), outputs);

  NNPR_ENSURE_STATUS(nnfw_close_session(session));

  // prepare result
  benchmark::Result result(t_model_load, t_prepare, t_execute, mp);

  // to stdout
  benchmark::printResult(result, (mp != nullptr));

  // to csv
  if (args.getWriteReport() == false)
    return 0;

  // prepare csv task
  std::string exec_basename;
  std::string nnpkg_basename;
  std::string backend_name = (available_backends) ? available_backends : default_backend_cand;
  {
    // I don't use PATH_MAX since it is not guaranteed value.
    // Instead, I've chosen smaller size than linux default 4096.
    char buf[1024];
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
