/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "args.h"

#include <nnfw_experimental.h>
#include <nnfw_internal.h>

#include <benchmark.h>
#include <misc/RandomGenerator.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

#include <libgen.h>

#define NNFW_ASSERT_FAIL(expr, msg)   \
  if ((expr) != NNFW_STATUS_NO_ERROR) \
  {                                   \
    throw std::runtime_error{msg};    \
  }

namespace
{

template <typename T>
void randomData(nnfw::misc::RandomGenerator &randgen, std::vector<uint8_t> &dest)
{
  size_t elements = dest.size() / sizeof(T);
  assert(dest.size() % sizeof(T) == 0);

  std::vector<T> vec(elements);
  for (uint64_t i = 0; i < elements; i++)
  {
    vec[i] = randgen.generate<T>();
  }
  memcpy(dest.data(), vec.data(), elements * sizeof(T));
}

void randomBoolData(nnfw::misc::RandomGenerator &randgen, std::vector<uint8_t> &dest)
{
  size_t elements = dest.size();
  std::vector<uint8_t> vec(elements);
  for (uint64_t i = 0; i < elements; i++)
  {
    bool value = randgen.generate<bool>();
    dest[i] = value ? 1 : 0;
  }
}

inline uint64_t num_elems(const nnfw_tensorinfo *ti)
{
  uint64_t n = 1;
  for (uint32_t i = 0; i < ti->rank; ++i)
  {
    n *= ti->dims[i];
  }
  return n;
}

inline size_t sizeOfNnfwType(NNFW_TYPE type)
{
  switch (type)
  {
    case NNFW_TYPE_TENSOR_BOOL:
    case NNFW_TYPE_TENSOR_UINT8:
    case NNFW_TYPE_TENSOR_QUANT8_ASYMM:
    case NNFW_TYPE_TENSOR_QUANT8_ASYMM_SIGNED:
      return 1;
    case NNFW_TYPE_TENSOR_QUANT16_SYMM_SIGNED:
      return 2;
    case NNFW_TYPE_TENSOR_FLOAT32:
    case NNFW_TYPE_TENSOR_INT32:
      return 4;
    case NNFW_TYPE_TENSOR_INT64:
      return 8;
    default:
      throw std::runtime_error{"Invalid tensor type"};
  }
}

static const char *default_backend_cand = "cpu";

} // namespace

int main(const int argc, char **argv)
{
  onert_run::Args args(argc, argv);

  std::chrono::milliseconds t_model_load(0), t_prepare(0);

  // TODO Apply verbose level to phases
  const int verbose = args.getVerboseLevel();
  benchmark::Phases phases(
    benchmark::PhaseOption{args.getMemoryPoll(), args.getGpuMemoryPoll(), args.getRunDelay()});

  nnfw_session *onert_session = nullptr;

  try
  {
    NNFW_ASSERT_FAIL(nnfw_create_session(&onert_session), "[ ERROR ] Failure to open session");

    phases.run("MODEL_LOAD", [&](const benchmark::Phase &, uint32_t) {
      NNFW_ASSERT_FAIL(
        nnfw_load_model_from_modelfile(onert_session, args.getModelFilename().c_str()),
        "[ ERROR ] Failure during model load");
    });

    uint32_t num_inputs = 0;
    uint32_t num_outputs = 0;
    NNFW_ASSERT_FAIL(nnfw_input_size(onert_session, &num_inputs),
                     "[ ERROR ] Failure during get model inputs");
    NNFW_ASSERT_FAIL(nnfw_output_size(onert_session, &num_outputs),
                     "[ ERROR ] Failure during get model outputs");

    if (args.getInputShapes().size() != 0)
    {
      const int dim_values = args.getInputShapes().size();
      int offset = 0;

      for (uint32_t i = 0; i < num_inputs; i++)
      {
        nnfw_tensorinfo ti_input;
        NNFW_ASSERT_FAIL(nnfw_input_tensorinfo(onert_session, i, &ti_input),
                         "[ ERROR ] Failure to get input info");

        for (uint32_t axis = 0; axis < ti_input.rank; axis++, offset++)
        {
          ti_input.dims[axis] =
            ((offset < dim_values) ? args.getInputShapes()[offset] : ti_input.dims[axis]);
        }

        NNFW_ASSERT_FAIL(nnfw_set_input_tensorinfo(onert_session, i, &ti_input),
                         "[ ERROR ] Failure to set input shape");

        if (offset >= dim_values)
          break;
      }
    }

    phases.run("PREPARE", [&](const benchmark::Phase &, uint32_t) {
      NNFW_ASSERT_FAIL(nnfw_prepare(onert_session), "[ ERROR ] Failure to compile");
    });

    // Load input from raw or dumped tensor file.
    // Two options are exclusive and will be checked from Args.
    if (!args.getInputFilename().empty())
    {
      throw std::runtime_error{"[ NYI ] load input file"};
    }

    // Prepare input/output data
    std::vector<std::vector<uint8_t>> inputs(num_inputs);
    std::vector<std::vector<uint8_t>> outputs(num_outputs);

    const int seed = 1; /* TODO Add an option for seed value */
    nnfw::misc::RandomGenerator randgen{seed, 0.0f, 2.0f};

    for (uint32_t i = 0; i < num_inputs; i++)
    {
      nnfw_tensorinfo ti_input;
      NNFW_ASSERT_FAIL(nnfw_input_tensorinfo(onert_session, i, &ti_input),
                       "[ ERROR ] Failure during get input data info");
      size_t input_size = num_elems(&ti_input) * sizeOfNnfwType(ti_input.dtype);

      inputs[i].resize(input_size);

      switch (ti_input.dtype)
      {
        case NNFW_TYPE_TENSOR_BOOL:
          randomBoolData(randgen, inputs[i]);
          break;
        case NNFW_TYPE_TENSOR_UINT8:
        case NNFW_TYPE_TENSOR_QUANT8_ASYMM:
          randomData<uint8_t>(randgen, inputs[i]);
          break;
        case NNFW_TYPE_TENSOR_QUANT8_ASYMM_SIGNED:
          randomData<int8_t>(randgen, inputs[i]);
          break;
        case NNFW_TYPE_TENSOR_QUANT16_SYMM_SIGNED:
          randomData<int16_t>(randgen, inputs[i]);
        case NNFW_TYPE_TENSOR_FLOAT32:
          randomData<float>(randgen, inputs[i]);
          break;
        case NNFW_TYPE_TENSOR_INT32:
          randomData<int32_t>(randgen, inputs[i]);
          break;
        case NNFW_TYPE_TENSOR_INT64:
          randomData<uint64_t>(randgen, inputs[i]);
          break;
        default:
          std::cerr << "[ ERROR ] "
                    << "Unspported input data type" << std::endl;
          exit(-1);
          break;
      }

      NNFW_ASSERT_FAIL(
        nnfw_set_input(onert_session, i, ti_input.dtype, inputs[i].data(), input_size),
        "[ ERROR ] Failure to set input tensor buffer");
    }

    for (uint32_t i = 0; i < num_outputs; i++)
    {
      nnfw_tensorinfo ti_output;
      NNFW_ASSERT_FAIL(nnfw_output_tensorinfo(onert_session, i, &ti_output),
                       "[ ERROR ] Failure during get output tensor info");

      uint64_t output_elements = num_elems(&ti_output);
      size_t output_size = output_elements * sizeOfNnfwType(ti_output.dtype);
      outputs[i].resize(output_size);

      NNFW_ASSERT_FAIL(
        nnfw_set_output(onert_session, i, ti_output.dtype, outputs[i].data(), output_size),
        "[ ERROR ] Failure to set output tensor buffer");
    }

    // NOTE: Measuring memory can't avoid taking overhead. Therefore, memory will be measured on the
    // only warmup.
    if (verbose == 0)
    {
      phases.run(
        "WARMUP",
        [&](const benchmark::Phase &, uint32_t) {
          NNFW_ASSERT_FAIL(nnfw_run(onert_session), "[Execution] Can't execute warmup");
        },
        args.getWarmupRuns());
      phases.run(
        "EXECUTE",
        [&](const benchmark::Phase &, uint32_t) {
          NNFW_ASSERT_FAIL(nnfw_run(onert_session), "[Execution] Can't execute");
        },
        args.getNumRuns(), true);
    }
    else
    {
      phases.run(
        "WARMUP",
        [&](const benchmark::Phase &, uint32_t) {
          NNFW_ASSERT_FAIL(nnfw_run(onert_session), "[Execution] Can't execute warmup");
        },
        [&](const benchmark::Phase &phase, uint32_t nth) {
          std::cout << "... "
                    << "warmup " << nth + 1 << " takes " << phase.time[nth] / 1e3 << " ms"
                    << std::endl;
        },
        args.getWarmupRuns());
      phases.run(
        "EXECUTE",
        [&](const benchmark::Phase &, uint32_t) {
          NNFW_ASSERT_FAIL(nnfw_run(onert_session), "[Execution] Can't execute");
        },
        [&](const benchmark::Phase &phase, uint32_t nth) {
          std::cout << "... "
                    << "run " << nth + 1 << " takes " << phase.time[nth] / 1e3 << " ms"
                    << std::endl;
        },
        args.getNumRuns(), true);
    }

    // prepare result
    benchmark::Result result(phases);

    // to stdout
    benchmark::printResult(result);

    if (args.getWriteReport())
    {
      // prepare csv task
      std::string exec_basename;
      std::string model_basename;
      std::string backend_name = default_backend_cand;
      {
        std::vector<char> vpath(args.getModelFilename().begin(), args.getModelFilename().end() + 1);
        model_basename = basename(vpath.data());
        size_t lastindex = model_basename.find_last_of(".");
        model_basename = model_basename.substr(0, lastindex);
        exec_basename = basename(argv[0]);
      }
      benchmark::writeResult(result, exec_basename, model_basename, backend_name);
    }

    if (!args.getDumpFilename().empty())
    {
      throw std::runtime_error{"[ NYI ] dump file"};
    }

    if (!args.getCompareFilename().empty())
    {
      throw std::runtime_error{"[ NYI ] compare file"};
    }
  }
  catch (const std::exception &e)
  {
    std::cerr << e.what() << '\n';
    return 1;
  }

  return 0;
}
