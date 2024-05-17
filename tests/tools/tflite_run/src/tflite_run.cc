/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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
#include "tensor_dumper.h"
#include "tensor_loader.h"
#include "misc/EnvVar.h"
#include "misc/fp32.h"
#include "tflite/Diff.h"
#include "tflite/Assert.h"
#include "tflite/Session.h"
#include "tflite/RandomInputInitializer.h"
#include "tflite/InterpreterSession.h"
#include "misc/tensor/IndexIterator.h"
#include "misc/tensor/Object.h"
#include "benchmark.h"

#include <iostream>
#include <chrono>
#include <algorithm>
#include <vector>

#include <libgen.h>

using namespace tflite;
using namespace nnfw::tflite;
using namespace std::placeholders; // for _1, _2 ...

namespace
{

void print_max_idx(float *f, int size)
{
  float *p = std::max_element(f, f + size);
  std::cout << "max:" << p - f;
}

static const char *default_backend_cand = "tflite_cpu";

} // namespace

int main(const int argc, char **argv)
{
  TFLiteRun::Args args(argc, argv);

  std::chrono::milliseconds t_model_load(0), t_prepare(0);

  // TODO Apply verbose level to phases
  const int verbose = args.getVerboseLevel();
  benchmark::Phases phases(
    benchmark::PhaseOption{args.getMemoryPoll(), args.getGpuMemoryPoll(), args.getRunDelay()});

  TfLiteModel *model = nullptr;

  try
  {
    phases.run("MODEL_LOAD", [&](const benchmark::Phase &, uint32_t) {
      model = TfLiteModelCreateFromFile(args.getTFLiteFilename().c_str());
    });
  }
  catch (const std::exception &e)
  {
    std::cerr << e.what() << '\n';
    return 1;
  }

  if (model == nullptr)
  {
    throw std::runtime_error{"Cannot create model"};
  }

  auto options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsSetNumThreads(options, nnfw::misc::EnvVar("THREAD").asInt(1));

  TfLiteInterpreter *interpreter = TfLiteInterpreterCreate(model, options);
  auto sess = std::make_shared<nnfw::tflite::InterpreterSession>(interpreter);
  try
  {
    phases.run("PREPARE", [&](const benchmark::Phase &, uint32_t) { sess->prepare(); });
  }
  catch (const std::exception &e)
  {
    std::cerr << e.what() << '\n';
    return 1;
  }

  if (args.getInputShapes().size() != 0)
  {
    const auto dim_values = args.getInputShapes().size();
    int32_t offset = 0;

    auto const input_count = TfLiteInterpreterGetInputTensorCount(interpreter);
    for (int32_t id = 0; id < input_count; id++)
    {
      TfLiteTensor *tensor = TfLiteInterpreterGetInputTensor(interpreter, id);
      std::vector<int32_t> new_dim;
      new_dim.resize(TfLiteTensorNumDims(tensor));

      for (int32_t axis = 0; axis < TfLiteTensorNumDims(tensor); axis++, offset++)
      {
        new_dim[axis] =
          ((offset < dim_values) ? args.getInputShapes()[offset] : TfLiteTensorDim(tensor, axis));
      }

      TfLiteInterpreterResizeInputTensor(interpreter, id, new_dim.data(), new_dim.size());

      if (offset >= dim_values)
        break;
    }
    TfLiteInterpreterAllocateTensors(interpreter);
  }

  TFLiteRun::TensorLoader tensor_loader(*interpreter);

  // Load input from raw or dumped tensor file.
  // Two options are exclusive and will be checked from Args.
  if (!args.getInputFilename().empty() || !args.getCompareFilename().empty())
  {
    if (!args.getInputFilename().empty())
    {
      tensor_loader.loadRawInputTensors(args.getInputFilename());
    }
    else
    {
      tensor_loader.loadDumpedTensors(args.getCompareFilename());
    }
  }
  else
  {
    const int seed = 1; /* TODO Add an option for seed value */
    nnfw::misc::RandomGenerator randgen{seed, 0.0f, 2.0f};

    RandomInputInitializer initializer{randgen};
    initializer.run(*interpreter);
  }

  TFLiteRun::TensorDumper tensor_dumper;
  // Must be called before `interpreter->Invoke()`
  tensor_dumper.addInputTensors(*interpreter);

  std::cout << "input tensor indices = [";
  auto const input_count = TfLiteInterpreterGetInputTensorCount(interpreter);
  for (int32_t idx = 0; idx < input_count; idx++)
  {
    std::cout << idx << ",";
  }
  std::cout << "]" << std::endl;

  // NOTE: Measuring memory can't avoid taking overhead. Therefore, memory will be measured on the
  // only warmup.
  if (verbose == 0)
  {
    phases.run(
      "WARMUP", [&](const benchmark::Phase &, uint32_t) { sess->run(); }, args.getWarmupRuns());
    phases.run(
      "EXECUTE", [&](const benchmark::Phase &, uint32_t) { sess->run(); }, args.getNumRuns(), true);
  }
  else
  {
    phases.run(
      "WARMUP", [&](const benchmark::Phase &, uint32_t) { sess->run(); },
      [&](const benchmark::Phase &phase, uint32_t nth) {
        std::cout << "... "
                  << "warmup " << nth + 1 << " takes " << phase.time[nth] / 1e3 << " ms"
                  << std::endl;
      },
      args.getWarmupRuns());
    phases.run(
      "EXECUTE", [&](const benchmark::Phase &, uint32_t) { sess->run(); },
      [&](const benchmark::Phase &phase, uint32_t nth) {
        std::cout << "... "
                  << "run " << nth + 1 << " takes " << phase.time[nth] / 1e3 << " ms" << std::endl;
      },
      args.getNumRuns(), true);
  }

  sess->teardown();

  // Must be called after `interpreter->Invoke()`
  tensor_dumper.addOutputTensors(*interpreter);

  std::cout << "output tensor indices = [";
  auto const output_count = TfLiteInterpreterGetOutputTensorCount(interpreter);
  for (int32_t idx = 0; idx < output_count; idx++)
  {
    auto tensor = TfLiteInterpreterGetOutputTensor(interpreter, idx);
    print_max_idx(reinterpret_cast<float *>(TfLiteTensorData(tensor)),
                  TfLiteTensorByteSize(tensor) / sizeof(float));

    std::cout << "),";
  }
  std::cout << "]" << std::endl;

  // TODO Apply verbose level to result

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
      std::vector<char> vpath(args.getTFLiteFilename().begin(), args.getTFLiteFilename().end() + 1);
      model_basename = basename(vpath.data());
      size_t lastindex = model_basename.find_last_of(".");
      model_basename = model_basename.substr(0, lastindex);
      exec_basename = basename(argv[0]);
    }
    benchmark::writeResult(result, exec_basename, model_basename, backend_name);
  }

  if (!args.getDumpFilename().empty())
  {
    const std::string &dump_filename = args.getDumpFilename();
    tensor_dumper.dump(dump_filename);
    std::cout << "Input/output tensors have been dumped to file \"" << dump_filename << "\"."
              << std::endl;
  }

  if (!args.getCompareFilename().empty())
  {
    const std::string &compare_filename = args.getCompareFilename();
    std::cout << "========================================" << std::endl;
    std::cout << "Comparing the results with \"" << compare_filename << "\"." << std::endl;
    std::cout << "========================================" << std::endl;

    // TODO Code duplication (copied from RandomTestRunner)

    int tolerance = nnfw::misc::EnvVar("TOLERANCE").asInt(1);

    auto equals = [tolerance](float lhs, float rhs) {
      // NOTE Hybrid approach
      // TODO Allow users to set tolerance for absolute_epsilon_equal
      if (nnfw::misc::fp32::absolute_epsilon_equal(lhs, rhs))
      {
        return true;
      }

      return nnfw::misc::fp32::epsilon_equal(lhs, rhs, tolerance);
    };

    nnfw::misc::tensor::Comparator comparator(equals);
    TfLiteInterpMatchApp app(comparator);
    bool res = true;

    for (int32_t idx = 0; idx < output_count; idx++)
    {
      auto expected = tensor_loader.getOutput(idx);
      auto const tensor = TfLiteInterpreterGetOutputTensor(interpreter, idx);
      auto obtained = nnfw::tflite::TensorView<float>::make(tensor);

      res = res && app.compareSingleTensorView(expected, obtained, idx);
    }

    if (!res)
    {
      return 255;
    }
  }

  return 0;
}
