/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "tensorflow/lite/model.h"
#include "tensorflow/lite/kernels/register.h"

#include "args.h"
#include "tensor_view.h"
#include "misc/EnvVar.h"
#include "misc/RandomGenerator.h"
#include "misc/tensor/IndexIterator.h"
#include "misc/tensor/Object.h"
#include "benchmark.h"

#include <iostream>
#include <chrono>
#include <algorithm>
#include <vector>
#include <memory>

using namespace std::placeholders; // for _1, _2 ...

#define TFLITE_ENSURE(exp)                                             \
  {                                                                    \
    const TfLiteStatus status = (exp);                                 \
                                                                       \
    if (status != kTfLiteOk)                                           \
    {                                                                  \
      std::ostringstream ss;                                           \
      ss << #exp << " failed (" << __FILE__ << ":" << __LINE__ << ")"; \
      throw std::runtime_error{ss.str()};                              \
    }                                                                  \
  }

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
  tflite::StderrReporter error_reporter;

  TFLiteRun220::Args args(argc, argv);

  std::chrono::milliseconds t_model_load(0), t_prepare(0);

  benchmark::Phases phases(benchmark::PhaseOption{args.getMemoryPoll(), args.getGpuMemoryPoll()});

  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<tflite::Interpreter> interpreter;
  std::unique_ptr<tflite::TfLiteVerifier> verifier{new BMFlatBufferVerifier};

  try
  {
    phases.run("MODEL_LOAD", [&](const benchmark::Phase &, uint32_t) {
      if (args.getModelValidate())
      {
        model = tflite::FlatBufferModel::VerifyAndBuildFromFile(args.getTFLiteFilename().c_str(),
                                                        verifier.get(), &error_reporter);
      }
      else
      {
        model = tflite::FlatBufferModel::BuildFromFile(args.getTFLiteFilename().c_str(), &error_reporter);
      }
      if (model == nullptr)
      {
        throw std::runtime_error{"Cannot create model"};
      }

      // Use tflite's resolver, not onert's one
      tflite::ops::builtin::BuiltinOpResolver resolver;
      tflite::InterpreterBuilder builder(*model, resolver);
      TFLITE_ENSURE(builder(&interpreter))
      interpreter->SetNumThreads(nnfw::misc::EnvVar("THREAD").asInt(-1));
    });
  }
  catch (const std::exception &e)
  {
    std::cerr << e.what() << '\n';
    return 1;
  }

  const bool use_nnapi = nnfw::misc::EnvVar("USE_NNAPI").asBool(false);

  try
  {
    phases.run("PREPARE", [&](const benchmark::Phase &, uint32_t) {
      interpreter->UseNNAPI(use_nnapi);
      interpreter->AllocateTensors();
    });
  }
  catch (const std::exception &e)
  {
    std::cerr << e.what() << '\n';
    return 1;
  }

  const int seed = 1; /* TODO Add an option for seed value */
  nnfw::misc::RandomGenerator randgen{seed, 0.0f, 2.0f};

  // No input specified. So we fill the input tensors with random values.
  for (const auto &o : interpreter->inputs())
  {
    TfLiteTensor *tensor = interpreter->tensor(o);
    if (tensor->type == kTfLiteInt32)
    {
      // Generate singed 32-bit integer (s32) input
      auto tensor_view = TFLiteRun220::TensorView<int32_t>::make(*interpreter, o);

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
      auto tensor_view = TFLiteRun220::TensorView<uint8_t>::make(*interpreter, o);

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
      auto tensor_view = TFLiteRun220::TensorView<bool>::make(*interpreter, o);

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

  std::cout << "input tensor indices = [";
  for (const auto &o : interpreter->inputs())
  {
    std::cout << o << ",";
  }
  std::cout << "]" << std::endl;

  // NOTE: Measuring memory can't avoid taking overhead. Therefore, memory will be measured on the
  // only warmup.

  // warmup runs
  phases.run("WARMUP", [&](const benchmark::Phase &, uint32_t) { interpreter->Invoke(); },
             [&](const benchmark::Phase &phase, uint32_t nth) {
               std::cout << "... "
                         << "warmup " << nth + 1 << " takes " << phase.time[nth] / 1e3 << " ms"
                         << std::endl;
             },
             args.getWarmupRuns());

  // actual runs
  phases.run("EXECUTE", [&](const benchmark::Phase &, uint32_t) { interpreter->Invoke(); },
             [&](const benchmark::Phase &phase, uint32_t nth) {
               std::cout << "... "
                         << "run " << nth + 1 << " takes " << phase.time[nth] / 1e3 << " ms"
                         << std::endl;
             },
             args.getNumRuns(), true);

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

  return 0;
}
