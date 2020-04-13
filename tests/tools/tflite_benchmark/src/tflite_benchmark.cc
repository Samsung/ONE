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

#include "tflite/Assert.h"
#include "tflite/Session.h"
#include "tflite/InterpreterSession.h"
#include "tflite/NNAPISession.h"
#include "tflite/Diff.h"
#include "misc/tensor/IndexIterator.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>

#include "misc/EnvVar.h"
#include "misc/benchmark.h"

using namespace tflite;
using namespace nnfw::tflite;

void help(std::ostream &out, const int argc, char **argv)
{
  std::string cmd = argv[0];
  auto pos = cmd.find_last_of("/");
  if (pos != std::string::npos)
    cmd = cmd.substr(pos + 1);

  out << "use:" << std::endl << cmd << " <model file name>" << std::endl;
}

bool checkParams(const int argc, char **argv)
{
  try
  {
    if (argc < 2)
    {
      help(std::cerr, argc, argv);
      return false;
    }
  }
  catch (const std::exception &e)
  {
    std::cerr << e.what() << std::endl;

    return false;
  }

  return true;
}

template <typename T> class Accumulator
{
public:
  Accumulator() = default;

  void operator()(T val)
  {
    ++_num;
    _sum += val;
    _log_sum += std::log(val);
    _min = std::min(_min, val);
    _max = std::max(_max, val);
  }

  T mean() const { return _sum / static_cast<T>(_num); }
  // Calculating geometric mean with logs
  //   "Geometric Mean of (V1, V2, ... Vn)"
  // = (V1*V2*...*Vn)^(1/n)
  // = exp(log((V1*V2*...*Vn)^(1/n)))
  // = exp(log((V1*V2*...*Vn)/n)))
  // = exp((log(V1) + log(V2) + ... + log(Vn))/n)
  // = exp(_log_sum/num)
  T geomean() const { return std::exp(_log_sum / static_cast<T>(_num)); }
  T min() const { return _min; }
  T max() const { return _max; }

private:
  uint32_t _num = 0u;
  T _sum = 0.0;
  T _log_sum = 0.0;
  T _min = std::numeric_limits<T>::max();
  T _max = std::numeric_limits<T>::lowest();
};

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

int main(const int argc, char **argv)
{

  if (!checkParams(argc, argv))
  {
    return -1;
  }

  const auto filename = argv[1];

  const bool use_nnapi = nnfw::misc::EnvVar("USE_NNAPI").asBool(false);
  const auto thread_count = nnfw::misc::EnvVar("THREAD").asInt(-1);
  const auto pause = nnfw::misc::EnvVar("PAUSE").asInt(0);
  const auto microsec = nnfw::misc::EnvVar("MICROSEC").asBool(0);

  std::cout << "Num threads: " << thread_count << std::endl;
  if (use_nnapi)
  {
    std::cout << "Use NNAPI" << std::endl;
  }

  assert(pause >= 0);
  if (pause > 0)
  {
    std::cout << "Insert " << pause << "s pause between iterations" << std::endl;
  }

  struct TimeUnit
  {
    const char *str;
    std::function<int64_t(int64_t)> conv;
  } tu = {"ms", [](int64_t v) { return v / 1000; }};

  if (microsec)
  {
    tu.str = "us";
    tu.conv = [](int64_t v) { return v; };
  }

  StderrReporter error_reporter;

  std::unique_ptr<tflite::TfLiteVerifier> verifier{new BMFlatBufferVerifier};

  auto model = FlatBufferModel::VerifyAndBuildFromFile(filename, verifier.get(), &error_reporter);
  if (model == nullptr)
  {
    std::cerr << "Cannot create model" << std::endl;
    return -1;
  }

  BuiltinOpResolver resolver;

  InterpreterBuilder builder(*model, resolver);

  std::unique_ptr<Interpreter> interpreter;

  try
  {
    TFLITE_ENSURE(builder(&interpreter));
  }
  catch (const std::exception &e)
  {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  // Show inputs
  for (uint32_t n = 0; n < interpreter->inputs().size(); ++n)
  {
    // TODO Print shape
    auto tensor_id = interpreter->inputs().at(n);
    auto tensor_ptr = interpreter->tensor(tensor_id);

    std::cout << "Input #" << n << ":" << std::endl;
    std::cout << "  Name: " << tensor_ptr->name << std::endl;
  }

  // Show outputs
  for (uint32_t n = 0; n < interpreter->outputs().size(); ++n)
  {
    // TODO Print shape
    auto tensor_id = interpreter->outputs().at(n);
    auto tensor_ptr = interpreter->tensor(tensor_id);

    std::cout << "Output #" << n << ":" << std::endl;
    std::cout << "  Name: " << tensor_ptr->name << std::endl;
  }

  interpreter->SetNumThreads(thread_count);

  std::shared_ptr<nnfw::tflite::Session> sess;

  if (use_nnapi)
  {
    sess = std::make_shared<nnfw::tflite::NNAPISession>(interpreter.get());
  }
  else
  {
    sess = std::make_shared<nnfw::tflite::InterpreterSession>(interpreter.get());
  }

  //
  // Warming-up
  //
  for (uint32_t n = 0; n < 3; ++n)
  {
    std::chrono::microseconds elapsed(0);

    sess->prepare();

    for (const auto &id : interpreter->inputs())
    {
      TfLiteTensor *tensor = interpreter->tensor(id);
      if (tensor->type == kTfLiteInt32)
      {
        // Generate singed 32-bit integer (s32) input
        auto tensor_view = nnfw::tflite::TensorView<int32_t>::make(*interpreter, id);

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
        auto tensor_view = nnfw::tflite::TensorView<uint8_t>::make(*interpreter, id);

        uint8_t value = 0;

        nnfw::misc::tensor::iterate(tensor_view.shape())
            << [&](const nnfw::misc::tensor::Index &ind) {
                 // TODO Generate random values
                 tensor_view.at(ind) = value;
                 value = (value + 1) & 0xFF;
               };
      }
      else
      {
        assert(tensor->type == kTfLiteFloat32);

        const int seed = 1; /* TODO Add an option for seed value */
        RandomGenerator randgen{seed, 0.0f, 0.2f};
        const float *end = reinterpret_cast<const float *>(tensor->data.raw_const + tensor->bytes);
        for (float *ptr = tensor->data.f; ptr < end; ptr++)
        {
          *ptr = randgen.generate<float>();
        }
      }
    }

    nnfw::misc::benchmark::measure(elapsed) << [&](void) {
      if (!sess->run())
      {
        assert(0 && "run failed");
      }
    };
    sess->teardown();

    std::cout << "Warming-up " << n << ": " << tu.conv(elapsed.count()) << tu.str << std::endl;
  }

  //
  // Measure
  //
  const auto cnt = nnfw::misc::EnvVar("COUNT").asInt(1);

  Accumulator<double> acc;

  for (int n = 0; n < cnt; ++n)
  {
    std::chrono::microseconds elapsed(0);

    sess->prepare();
    nnfw::misc::benchmark::measure(elapsed) << [&](void) {
      if (!sess->run())
      {
        assert(0 && "run failed");
      }
    };
    sess->teardown();

    acc(elapsed.count());

    std::cout << "Iteration " << n << ": " << tu.conv(elapsed.count()) << tu.str << std::endl;

    // Insert "pause"
    if ((n != cnt - 1) && (pause > 0))
    {
      std::this_thread::sleep_for(std::chrono::seconds(pause));
    }
  }

  auto v_min = tu.conv(acc.min());
  auto v_max = tu.conv(acc.max());
  auto v_mean = tu.conv(acc.mean());
  auto v_geomean = tu.conv(acc.geomean());

  std::cout << "--------" << std::endl;
  std::cout << "Min: " << v_min << tu.str << std::endl;
  std::cout << "Max: " << v_max << tu.str << std::endl;
  std::cout << "Mean: " << v_mean << tu.str << std::endl;
  std::cout << "GeoMean: " << v_geomean << tu.str << std::endl;

  return 0;
}
