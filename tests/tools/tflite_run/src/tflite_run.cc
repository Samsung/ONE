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

#include "tflite/ext/kernels/register.h"
#include "tensorflow/lite/model.h"

#include "args.h"
#include "tensor_dumper.h"
#include "tensor_loader.h"
#include "misc/benchmark.h"
#include "misc/EnvVar.h"
#include "misc/fp32.h"
#include "tflite/Diff.h"
#include "tflite/Assert.h"
#include "tflite/Session.h"
#include "tflite/InterpreterSession.h"
#include "tflite/NNAPISession.h"
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

// Verifies whether the model is a flatbuffer file.
class BMFlatBufferVerifier : public tflite::TfLiteVerifier
{
public:
  bool Verify(const char *data, int length, tflite::ErrorReporter *reporter) override
  {

    flatbuffers::Verifier verifier(reinterpret_cast<const uint8_t *>(data), length);
    if (!tflite::VerifyModelBuffer(verifier))
    {
      reporter->Report("The model is not a valid Flatbuffer file");
      return false;
    }
    return true;
  }
};

} // namespace anonymous

int main(const int argc, char **argv)
{
  const bool use_nnapi = nnfw::misc::EnvVar("USE_NNAPI").asBool(false);

  StderrReporter error_reporter;

  TFLiteRun::Args args(argc, argv);

  std::chrono::milliseconds t_model_load(0), t_prepare(0);

  benchmark::Phases phases(
      benchmark::PhaseOption{args.getMemoryPoll(), args.getGpuMemoryPoll(), args.getRunDelay()});

  std::unique_ptr<FlatBufferModel> model;
  std::unique_ptr<Interpreter> interpreter;
  std::unique_ptr<tflite::TfLiteVerifier> verifier{new BMFlatBufferVerifier};

  try
  {
    phases.run("MODEL_LOAD", [&](const benchmark::Phase &, uint32_t) {
      if (args.getModelValidate())
      {
        model = FlatBufferModel::VerifyAndBuildFromFile(args.getTFLiteFilename().c_str(),
                                                        verifier.get(), &error_reporter);
      }
      else
      {
        model = FlatBufferModel::BuildFromFile(args.getTFLiteFilename().c_str(), &error_reporter);
      }
      if (model == nullptr)
      {
        throw std::runtime_error{"Cannot create model"};
      }

      BuiltinOpResolver resolver;
      InterpreterBuilder builder(*model, resolver);
      TFLITE_ENSURE(builder(&interpreter))
      interpreter->SetNumThreads(nnfw::misc::EnvVar("THREAD").asInt(-1));
    });
  }
  catch (const std::exception &e)
  {
    std::cerr << e.what() << '\n';
    return 1;
  }

  std::shared_ptr<nnfw::tflite::Session> sess;

  if (use_nnapi)
  {
    sess = std::make_shared<nnfw::tflite::NNAPISession>(interpreter.get());
  }
  else
  {
    sess = std::make_shared<nnfw::tflite::InterpreterSession>(interpreter.get());
  }

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
    const int dim_values = args.getInputShapes().size();
    int offset = 0;

    for (const auto &id : interpreter->inputs())
    {
      TfLiteTensor *tensor = interpreter->tensor(id);
      std::vector<int32_t> new_dim;
      new_dim.resize(tensor->dims->size);

      for (uint32_t axis = 0; axis < tensor->dims->size; axis++, offset++)
      {
        new_dim[axis] =
            ((offset < dim_values) ? args.getInputShapes()[offset] : tensor->dims->data[axis]);
      }

      interpreter->ResizeInputTensor(id, new_dim);

      if (offset >= dim_values)
        break;
    }
    interpreter->AllocateTensors();
  }

  TFLiteRun::TensorLoader tensor_loader(*interpreter);

  // Load input from raw or dumped tensor file.
  // Two options are exclusive and will be checked from Args.
  if (!args.getInputFilename().empty() || !args.getCompareFilename().empty())
  {
    if (!args.getInputFilename().empty())
    {
      tensor_loader.loadRawTensors(args.getInputFilename(), interpreter->inputs());
    }
    else
    {
      tensor_loader.loadDumpedTensors(args.getCompareFilename());
    }

    for (const auto &o : interpreter->inputs())
    {
      const auto &tensor_view = tensor_loader.get(o);
      TfLiteTensor *tensor = interpreter->tensor(o);

      memcpy(reinterpret_cast<void *>(tensor->data.f),
             reinterpret_cast<const void *>(tensor_view._base), tensor->bytes);
    }
  }
  else
  {
    const int seed = 1; /* TODO Add an option for seed value */
    nnfw::misc::RandomGenerator randgen{seed, 0.0f, 2.0f};

    // No input specified. So we fill the input tensors with random values.
    for (const auto &o : interpreter->inputs())
    {
      TfLiteTensor *tensor = interpreter->tensor(o);
      if (tensor->type == kTfLiteInt32)
      {
        // Generate singed 32-bit integer (s32) input
        auto tensor_view = nnfw::tflite::TensorView<int32_t>::make(*interpreter, o);

        int32_t value = 0;

        nnfw::misc::tensor::iterate(tensor_view.shape())
            << [&](const nnfw::misc::tensor::Index &ind) {
                 // TODO Generate random values
                 // Gather operation: index should be within input coverage.
                 tensor_view.at(ind) = value;
                 value++;
               };
      }
      else if (tensor->type == kTfLiteUInt8)
      {
        // Generate unsigned 8-bit integer input
        auto tensor_view = nnfw::tflite::TensorView<uint8_t>::make(*interpreter, o);

        uint8_t value = 0;

        nnfw::misc::tensor::iterate(tensor_view.shape())
            << [&](const nnfw::misc::tensor::Index &ind) {
                 // TODO Generate random values
                 tensor_view.at(ind) = value;
                 value = (value + 1) & 0xFF;
               };
      }
      else if (tensor->type == kTfLiteBool)
      {
        // Generate bool input
        auto tensor_view = nnfw::tflite::TensorView<bool>::make(*interpreter, o);

        auto fp = static_cast<bool (nnfw::misc::RandomGenerator::*)(
            const ::nnfw::misc::tensor::Shape &, const ::nnfw::misc::tensor::Index &)>(
            &nnfw::misc::RandomGenerator::generate<bool>);
        const nnfw::misc::tensor::Object<bool> data(tensor_view.shape(),
                                                    std::bind(fp, randgen, _1, _2));

        nnfw::misc::tensor::iterate(tensor_view.shape())
            << [&](const nnfw::misc::tensor::Index &ind) {
                 const auto value = data.at(ind);
                 tensor_view.at(ind) = value;
               };
      }
      else
      {
        assert(tensor->type == kTfLiteFloat32);

        const float *end = reinterpret_cast<const float *>(tensor->data.raw_const + tensor->bytes);
        for (float *ptr = tensor->data.f; ptr < end; ptr++)
        {
          *ptr = randgen.generate<float>();
        }
      }
    }
  }

  TFLiteRun::TensorDumper tensor_dumper;
  // Must be called before `interpreter->Invoke()`
  tensor_dumper.addTensors(*interpreter, interpreter->inputs());

  std::cout << "input tensor indices = [";
  for (const auto &o : interpreter->inputs())
  {
    std::cout << o << ",";
  }
  std::cout << "]" << std::endl;

  // NOTE: Measuring memory can't avoid taking overhead. Therefore, memory will be measured on the
  // only warmup.
  // warmup runs
  phases.run("WARMUP", [&](const benchmark::Phase &, uint32_t) { sess->run(); },
             [&](const benchmark::Phase &phase, uint32_t nth) {
               std::cout << "... "
                         << "warmup " << nth + 1 << " takes " << phase.time[nth] / 1e3 << " ms"
                         << std::endl;
             },
             args.getWarmupRuns());

  // actual runs
  phases.run("EXECUTE", [&](const benchmark::Phase &, uint32_t) { sess->run(); },
             [&](const benchmark::Phase &phase, uint32_t nth) {
               std::cout << "... "
                         << "run " << nth + 1 << " takes " << phase.time[nth] / 1e3 << " ms"
                         << std::endl;
             },
             args.getNumRuns(), true);

  sess->teardown();

  // Must be called after `interpreter->Invoke()`
  tensor_dumper.addTensors(*interpreter, interpreter->outputs());

  std::cout << "output tensor indices = [";
  for (const auto &o : interpreter->outputs())
  {
    std::cout << o << "(";

    print_max_idx(interpreter->tensor(o)->data.f, interpreter->tensor(o)->bytes / sizeof(float));

    std::cout << "),";
  }
  std::cout << "]" << std::endl;

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

    for (const auto &o : interpreter->outputs())
    {
      auto expected = tensor_loader.get(o);
      auto obtained = nnfw::tflite::TensorView<float>::make(*interpreter, o);

      res = res && app.compareSingleTensorView(expected, obtained, o);
    }

    if (!res)
    {
      return 255;
    }
  }

  return 0;
}
