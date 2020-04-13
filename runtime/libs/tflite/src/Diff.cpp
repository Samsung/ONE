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

#include "tflite/Diff.h"
#include "tflite/ext/nnapi_delegate.h"

#include "misc/fp32.h"

#include "misc/tensor/IndexIterator.h"
#include "misc/tensor/IndexFormatter.h"
#include "misc/tensor/Zipper.h"
#include "misc/tensor/Comparator.h"

#include "misc/EnvVar.h"

#include <iostream>
#include <cassert>

class DiffSummary : public nnfw::misc::tensor::Comparator::Observer
{
public:
  DiffSummary()
      : max_abs_diff_index(0), max_abs_diff_expected{0.0f}, max_abs_diff_obtained{0.0f},
        max_abs_diff_value{0.0f}, max_rel_diff_index(0), max_rel_diff_expected{0.0f},
        max_rel_diff_obtained{0.0f}, max_rel_diff_value{0.0f}
  {
    // DO NOTHING
  }

public:
  void notify(const nnfw::misc::tensor::Index &index, float expected, float obtained) override;

public:
  nnfw::misc::tensor::Index max_abs_diff_index;
  float max_abs_diff_expected;
  float max_abs_diff_obtained;
  float max_abs_diff_value;

  nnfw::misc::tensor::Index max_rel_diff_index;
  float max_rel_diff_expected;
  float max_rel_diff_obtained;
  float max_rel_diff_value;
};

void DiffSummary::notify(const nnfw::misc::tensor::Index &index, float expected, float obtained)
{
  const auto abs_diff_value = std::fabs(expected - obtained);

  if (max_abs_diff_value < abs_diff_value)
  {
    max_abs_diff_index = index;
    max_abs_diff_value = abs_diff_value;
    max_abs_diff_expected = expected;
    max_abs_diff_obtained = obtained;
  }

  const auto rel_diff_value = nnfw::misc::fp32::relative_diff(expected, obtained);

  if (max_rel_diff_value < rel_diff_value)
  {
    max_rel_diff_index = index;
    max_rel_diff_value = rel_diff_value;
    max_rel_diff_expected = expected;
    max_rel_diff_obtained = obtained;
  }
}

template <typename T>
bool TfLiteInterpMatchApp::compareSingleTensorView(const nnfw::tflite::TensorView<T> &expected,
                                                   const nnfw::tflite::TensorView<T> &obtained,
                                                   int id) const
{
  std::vector<nnfw::misc::tensor::Diff<T>> diffs;
  assert(expected.shape() == obtained.shape());

  using nnfw::misc::tensor::zip;
  using nnfw::misc::tensor::Index;

  zip(expected.shape(), expected, obtained)
      << [&](const Index &index, T expected_value, T obtained_value) {
           if (expected_value != obtained_value)
           {
             diffs.emplace_back(index, expected_value, obtained_value);
           }
         };

  // TODO Unify summary generation code
  if (diffs.size() == 0)
  {
    std::cout << "  Tensor #" << id << ": MATCHED" << std::endl;
  }
  else
  {
    std::cout << "  Tensor #" << id << ": UNMATCHED" << std::endl;
    std::cout << "    " << diffs.size() << " diffs are detected" << std::endl;
  }

  if (diffs.size() > 0 && _verbose != 0)
  {
    std::cout << "    ---- Details ---" << std::endl;
    for (const auto &diff : diffs)
    {
      std::cout << "    Diff at [" << nnfw::misc::tensor::IndexFormatter(diff.index) << "]"
                << std::endl;
      std::cout << "      expected: " << diff.expected << std::endl;
      std::cout << "      obtained: " << diff.obtained << std::endl;
    }
  }

  return diffs.size() == 0;
}

template <>
bool TfLiteInterpMatchApp::compareSingleTensorView<float>(
    const nnfw::tflite::TensorView<float> &expected,
    const nnfw::tflite::TensorView<float> &obtained, int id) const
{
  DiffSummary summary;

  assert(expected.shape() == obtained.shape());
  auto diffs = _comparator.compare(expected.shape(), expected, obtained, &summary);

  // TODO Unify summary generation code
  if (diffs.size() == 0)
  {
    std::cout << "  Tensor #" << id << ": MATCHED" << std::endl;
  }
  else
  {
    std::cout << "  Tensor #" << id << ": UNMATCHED" << std::endl;
    std::cout << "    " << diffs.size() << " diffs are detected" << std::endl;
  }

  // Print out max_diff
  if (summary.max_abs_diff_value > 0)
  {
    std::cout << "    Max absolute diff at ["
              << nnfw::misc::tensor::IndexFormatter(summary.max_abs_diff_index) << "]" << std::endl;
    std::cout << "       expected: " << summary.max_abs_diff_expected << std::endl;
    std::cout << "       obtained: " << summary.max_abs_diff_obtained << std::endl;
    std::cout << "       absolute diff: " << summary.max_abs_diff_value << std::endl;
  }

  if (summary.max_rel_diff_value > 0)
  {
    const auto tolerance_level = summary.max_rel_diff_value / FLT_EPSILON;

    std::cout << "    Max relative diff at ["
              << nnfw::misc::tensor::IndexFormatter(summary.max_rel_diff_index) << "]" << std::endl;
    std::cout << "       expected: " << summary.max_rel_diff_expected << std::endl;
    std::cout << "       obtained: " << summary.max_rel_diff_obtained << std::endl;
    std::cout << "       relative diff: " << summary.max_rel_diff_value << std::endl;
    std::cout << "         (tolerance level = " << tolerance_level << ")" << std::endl;
  }

  if (diffs.size() > 0)
  {
    if (_verbose != 0)
    {
      std::cout << "    ---- Details ---" << std::endl;
      for (const auto &diff : diffs)
      {
        const auto absolute_diff = std::fabs(diff.expected - diff.obtained);
        const auto relative_diff = nnfw::misc::fp32::relative_diff(diff.expected, diff.obtained);
        const auto tolerance_level = relative_diff / FLT_EPSILON;

        std::cout << "    Diff at [" << nnfw::misc::tensor::IndexFormatter(diff.index) << "]"
                  << std::endl;
        std::cout << "      expected: " << diff.expected << std::endl;
        std::cout << "      obtained: " << diff.obtained << std::endl;
        std::cout << "      absolute diff: " << absolute_diff << std::endl;
        std::cout << "      relative diff: " << relative_diff << std::endl;
        std::cout << "         (tolerance level = " << tolerance_level << ")" << std::endl;
      }
    }

    return false;
  }
  return true;
}

#include <map>

bool TfLiteInterpMatchApp::run(::tflite::Interpreter &interp, ::tflite::Interpreter &nnapi) const
{
  assert(interp.outputs() == nnapi.outputs());

  bool all_matched = true;

  using Comparator = std::function<bool(int id, ::tflite::Interpreter &, ::tflite::Interpreter &)>;

  std::map<TfLiteType, Comparator> comparators;

  comparators[kTfLiteUInt8] = [this](int id, ::tflite::Interpreter &interp,
                                     ::tflite::Interpreter &nnapi) {
    const auto expected = nnfw::tflite::TensorView<uint8_t>::make(interp, id);
    const auto obtained = nnfw::tflite::TensorView<uint8_t>::make(nnapi, id);

    return compareSingleTensorView(expected, obtained, id);
  };

  comparators[kTfLiteInt32] = [this](int id, ::tflite::Interpreter &interp,
                                     ::tflite::Interpreter &nnapi) {
    const auto expected = nnfw::tflite::TensorView<int32_t>::make(interp, id);
    const auto obtained = nnfw::tflite::TensorView<int32_t>::make(nnapi, id);

    return compareSingleTensorView(expected, obtained, id);
  };

  comparators[kTfLiteFloat32] = [this](int id, ::tflite::Interpreter &interp,
                                       ::tflite::Interpreter &nnapi) {
    const auto expected = nnfw::tflite::TensorView<float>::make(interp, id);
    const auto obtained = nnfw::tflite::TensorView<float>::make(nnapi, id);

    return compareSingleTensorView(expected, obtained, id);
  };

  comparators[kTfLiteBool] = [this](int id, ::tflite::Interpreter &interp,
                                    ::tflite::Interpreter &nnapi) {
    const auto expected = nnfw::tflite::TensorView<bool>::make(interp, id);
    const auto obtained = nnfw::tflite::TensorView<bool>::make(nnapi, id);

    return compareSingleTensorView(expected, obtained, id);
  };

  for (const auto &id : interp.outputs())
  {
    assert(interp.tensor(id)->type == nnapi.tensor(id)->type);

    auto it = comparators.find(interp.tensor(id)->type);

    if (it == comparators.end())
    {
      throw std::runtime_error{"Not supported output type"};
    }

    const auto &comparator = it->second;

    if (!comparator(id, interp, nnapi))
    {
      all_matched = false;
    }
  }

  return all_matched;
}

#include "misc/tensor/Object.h"

using namespace std::placeholders;

template <> uint8_t RandomGenerator::generate<uint8_t>(void)
{
  // The value of type_range is 255.
  float type_range = static_cast<float>(std::numeric_limits<uint8_t>::max()) -
                     static_cast<float>(std::numeric_limits<uint8_t>::min());
  // Most _dist values range from -5.0 to 5.0.
  float min_range = -5.0f;
  float max_range = 5.0f;
  // NOTE shifted_relative_val has Gaussian distribution that origin mean was 0 and standard
  // deviation was 2. And then its values are distributed and shift to that mean is 127.5 and range
  // is about [0, 255].
  float shifted_relative_val = (_dist(_rand) - min_range) * type_range / (max_range - min_range);

  // shifted_relative_val is adjusted to be mapped to end points of the range, if it is out of range
  // values.
  if (shifted_relative_val < 0.0f)
  {
    return 0;
  }
  else if (shifted_relative_val > type_range)
  {
    return 255;
  }

  // Convert shifted_relative_val from float to uint8
  return static_cast<uint8_t>(shifted_relative_val);
}

template <> bool RandomGenerator::generate<bool>(void)
{
  std::uniform_int_distribution<> dist(0, 1); // [0, 1]
  return dist(_rand);
}

template <> int32_t RandomGenerator::generate<int32_t>(void)
{
  // Instead of INT_MAX, 4096 is chosen because int32_t input does not mean
  // that the model can have any value in int32_t can hold.
  // For example, one_hot operation gets indices as int32_t tensor.
  // However, we usually expect it would hold a value in [0..depth).
  // In our given model, depth was 10137.
  const int int32_random_max = 4096;
  std::uniform_int_distribution<> dist(0, int32_random_max);
  return dist(_rand);
}

#include "tflite/TensorLogger.h"
//
// Random Test Runner
//
int RandomTestRunner::run(const nnfw::tflite::Builder &builder)
{
  auto tfl_interp = builder.build();
  auto nnapi = builder.build();

  tfl_interp->UseNNAPI(false);

  // Allocate Tensors
  tfl_interp->AllocateTensors();
  nnapi->AllocateTensors();

  assert(tfl_interp->inputs() == nnapi->inputs());

  using ::tflite::Interpreter;
  using Initializer = std::function<void(int id, Interpreter *, Interpreter *)>;

  std::map<TfLiteType, Initializer> initializers;
  std::map<TfLiteType, Initializer> reseters;

  // Generate singed 32-bit integer (s32) input
  initializers[kTfLiteInt32] = [&](int id, Interpreter *tfl_interp, Interpreter *nnapi) {
    assert(tfl_interp->tensor(id)->type == kTfLiteInt32);
    assert(nnapi->tensor(id)->type == kTfLiteInt32);

    auto tfl_interp_view = nnfw::tflite::TensorView<int32_t>::make(*tfl_interp, id);
    auto nnapi_view = nnfw::tflite::TensorView<int32_t>::make(*nnapi, id);

    assert(tfl_interp_view.shape() == nnapi_view.shape());

    int32_t value = 0;

    nnfw::misc::tensor::iterate(tfl_interp_view.shape())
        << [&](const nnfw::misc::tensor::Index &ind) {
             // TODO Generate random values
             tfl_interp_view.at(ind) = value;
             nnapi_view.at(ind) = value;
             ++value;
           };
  };

  // Generate singed 32-bit integer (s32) input
  reseters[kTfLiteInt32] = [&](int id, Interpreter *tfl_interp, Interpreter *nnapi) {
    assert(tfl_interp->tensor(id)->type == kTfLiteInt32);
    assert(nnapi->tensor(id)->type == kTfLiteInt32);

    auto tfl_interp_view = nnfw::tflite::TensorView<int32_t>::make(*tfl_interp, id);
    auto nnapi_view = nnfw::tflite::TensorView<int32_t>::make(*nnapi, id);

    assert(tfl_interp_view.shape() == nnapi_view.shape());

    int32_t value = 0;

    nnfw::misc::tensor::iterate(tfl_interp_view.shape())
        << [&](const nnfw::misc::tensor::Index &ind) {
             // TODO Generate random values
             tfl_interp_view.at(ind) = value;
             nnapi_view.at(ind) = value;
           };
  };

  initializers[kTfLiteUInt8] = [&](int id, Interpreter *tfl_interp, Interpreter *nnapi) {
    assert(tfl_interp->tensor(id)->type == kTfLiteUInt8);
    assert(nnapi->tensor(id)->type == kTfLiteUInt8);

    auto tfl_interp_view = nnfw::tflite::TensorView<uint8_t>::make(*tfl_interp, id);
    auto nnapi_view = nnfw::tflite::TensorView<uint8_t>::make(*nnapi, id);

    assert(tfl_interp_view.shape() == nnapi_view.shape());

    auto fp = static_cast<uint8_t (RandomGenerator::*)(const ::nnfw::misc::tensor::Shape &,
                                                       const ::nnfw::misc::tensor::Index &)>(
        &RandomGenerator::generate<uint8_t>);
    const nnfw::misc::tensor::Object<uint8_t> data(tfl_interp_view.shape(),
                                                   std::bind(fp, _randgen, _1, _2));
    assert(tfl_interp_view.shape() == data.shape());

    nnfw::misc::tensor::iterate(tfl_interp_view.shape())
        << [&](const nnfw::misc::tensor::Index &ind) {
             const auto value = data.at(ind);

             tfl_interp_view.at(ind) = value;
             nnapi_view.at(ind) = value;
           };
  };

  reseters[kTfLiteUInt8] = [&](int id, Interpreter *tfl_interp, Interpreter *nnapi) {
    assert(tfl_interp->tensor(id)->type == kTfLiteUInt8);
    assert(nnapi->tensor(id)->type == kTfLiteUInt8);

    auto tfl_interp_view = nnfw::tflite::TensorView<uint8_t>::make(*tfl_interp, id);
    auto nnapi_view = nnfw::tflite::TensorView<uint8_t>::make(*nnapi, id);

    assert(tfl_interp_view.shape() == nnapi_view.shape());

    auto fp = static_cast<uint8_t (RandomGenerator::*)(const ::nnfw::misc::tensor::Shape &,
                                                       const ::nnfw::misc::tensor::Index &)>(
        &RandomGenerator::generate<uint8_t>);
    const nnfw::misc::tensor::Object<uint8_t> data(tfl_interp_view.shape(),
                                                   std::bind(fp, _randgen, _1, _2));
    assert(tfl_interp_view.shape() == data.shape());

    uint8_t value = 0;

    nnfw::misc::tensor::iterate(tfl_interp_view.shape())
        << [&](const nnfw::misc::tensor::Index &ind) {
             tfl_interp_view.at(ind) = value;
             nnapi_view.at(ind) = value;
           };
  };

  initializers[kTfLiteFloat32] = [&](int id, Interpreter *tfl_interp, Interpreter *nnapi) {
    assert(tfl_interp->tensor(id)->type == kTfLiteFloat32);
    assert(nnapi->tensor(id)->type == kTfLiteFloat32);

    auto tfl_interp_view = nnfw::tflite::TensorView<float>::make(*tfl_interp, id);
    auto nnapi_view = nnfw::tflite::TensorView<float>::make(*nnapi, id);

    assert(tfl_interp_view.shape() == nnapi_view.shape());

    auto fp = static_cast<float (RandomGenerator::*)(const ::nnfw::misc::tensor::Shape &,
                                                     const ::nnfw::misc::tensor::Index &)>(
        &RandomGenerator::generate<float>);
    const nnfw::misc::tensor::Object<float> data(tfl_interp_view.shape(),
                                                 std::bind(fp, _randgen, _1, _2));

    assert(tfl_interp_view.shape() == data.shape());

    nnfw::misc::tensor::iterate(tfl_interp_view.shape())
        << [&](const nnfw::misc::tensor::Index &ind) {
             const auto value = data.at(ind);

             tfl_interp_view.at(ind) = value;
             nnapi_view.at(ind) = value;
           };
  };

  reseters[kTfLiteFloat32] = [&](int id, Interpreter *tfl_interp, Interpreter *nnapi) {
    assert(tfl_interp->tensor(id)->type == kTfLiteFloat32);
    assert(nnapi->tensor(id)->type == kTfLiteFloat32);

    auto tfl_interp_view = nnfw::tflite::TensorView<float>::make(*tfl_interp, id);
    auto nnapi_view = nnfw::tflite::TensorView<float>::make(*nnapi, id);

    assert(tfl_interp_view.shape() == nnapi_view.shape());

    auto fp = static_cast<float (RandomGenerator::*)(const ::nnfw::misc::tensor::Shape &,
                                                     const ::nnfw::misc::tensor::Index &)>(
        &RandomGenerator::generate<float>);
    const nnfw::misc::tensor::Object<float> data(tfl_interp_view.shape(),
                                                 std::bind(fp, _randgen, _1, _2));

    assert(tfl_interp_view.shape() == data.shape());

    float value = 0;

    nnfw::misc::tensor::iterate(tfl_interp_view.shape())
        << [&](const nnfw::misc::tensor::Index &ind) {
             tfl_interp_view.at(ind) = value;
             nnapi_view.at(ind) = value;
           };
  };

  initializers[kTfLiteBool] = [&](int id, Interpreter *tfl_interp, Interpreter *nnapi) {
    assert(tfl_interp->tensor(id)->type == kTfLiteBool);
    assert(nnapi->tensor(id)->type == kTfLiteBool);

    auto tfl_interp_view = nnfw::tflite::TensorView<bool>::make(*tfl_interp, id);
    auto nnapi_view = nnfw::tflite::TensorView<bool>::make(*nnapi, id);

    assert(tfl_interp_view.shape() == nnapi_view.shape());

    auto fp = static_cast<bool (RandomGenerator::*)(const ::nnfw::misc::tensor::Shape &,
                                                    const ::nnfw::misc::tensor::Index &)>(
        &RandomGenerator::generate<bool>);
    const nnfw::misc::tensor::Object<bool> data(tfl_interp_view.shape(),
                                                std::bind(fp, _randgen, _1, _2));

    assert(tfl_interp_view.shape() == data.shape());

    nnfw::misc::tensor::iterate(tfl_interp_view.shape())
        << [&](const nnfw::misc::tensor::Index &ind) {
             const auto value = data.at(ind);

             tfl_interp_view.at(ind) = value;
             nnapi_view.at(ind) = value;
           };
  };

  reseters[kTfLiteBool] = [&](int id, Interpreter *tfl_interp, Interpreter *nnapi) {
    assert(tfl_interp->tensor(id)->type == kTfLiteBool);
    assert(nnapi->tensor(id)->type == kTfLiteBool);

    auto tfl_interp_view = nnfw::tflite::TensorView<bool>::make(*tfl_interp, id);
    auto nnapi_view = nnfw::tflite::TensorView<bool>::make(*nnapi, id);

    assert(tfl_interp_view.shape() == nnapi_view.shape());

    auto fp = static_cast<bool (RandomGenerator::*)(const ::nnfw::misc::tensor::Shape &,
                                                    const ::nnfw::misc::tensor::Index &)>(
        &RandomGenerator::generate<bool>);
    const nnfw::misc::tensor::Object<bool> data(tfl_interp_view.shape(),
                                                std::bind(fp, _randgen, _1, _2));

    assert(tfl_interp_view.shape() == data.shape());

    bool value = false;

    nnfw::misc::tensor::iterate(tfl_interp_view.shape())
        << [&](const nnfw::misc::tensor::Index &ind) {
             tfl_interp_view.at(ind) = value;
             nnapi_view.at(ind) = value;
           };
  };

  // Fill IFM with random numbers
  for (const auto id : tfl_interp->inputs())
  {
    assert(tfl_interp->tensor(id)->type == nnapi->tensor(id)->type);

    auto it = initializers.find(tfl_interp->tensor(id)->type);

    if (it == initializers.end())
    {
      throw std::runtime_error{"Not supported input type"};
    }

    it->second(id, tfl_interp.get(), nnapi.get());
  }

  // Fill OFM with 0
  for (const auto id : tfl_interp->outputs())
  {
    assert(tfl_interp->tensor(id)->type == nnapi->tensor(id)->type);

    auto it = reseters.find(tfl_interp->tensor(id)->type);

    if (it == reseters.end())
    {
      throw std::runtime_error{"Not supported input type"};
    }

    it->second(id, tfl_interp.get(), nnapi.get());
  }

  std::cout << "[NNAPI TEST] Run T/F Lite Interpreter without NNAPI" << std::endl;
  tfl_interp->Invoke();

  std::cout << "[NNAPI TEST] Run T/F Lite Interpreter with NNAPI" << std::endl;

  char *env = getenv("UPSTREAM_DELEGATE");

  if (env && !std::string(env).compare("1"))
  {
    nnapi->UseNNAPI(true);
    nnapi->Invoke();
  }
  else
  {
    nnfw::tflite::NNAPIDelegate d;

    // WARNING
    // primary_subgraph: Experimental interface. Return 1st sugbraph
    if (d.BuildGraph(&nnapi.get()->primary_subgraph()))
    {
      throw std::runtime_error{"Failed to BuildGraph"};
    }

    if (d.Invoke(&nnapi.get()->primary_subgraph()))
    {
      throw std::runtime_error{"Failed to BuildGraph"};
    }
  }

  // Compare OFM
  std::cout << "[NNAPI TEST] Compare the result" << std::endl;

  const auto tolerance = _param.tolerance;

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

  app.verbose() = _param.verbose;

  bool res = app.run(*tfl_interp, *nnapi);

  if (!res)
  {
    return 255;
  }

  std::cout << "[NNAPI TEST] PASSED" << std::endl;

  if (_param.tensor_logging)
    nnfw::tflite::TensorLogger::get().save(_param.log_path, *tfl_interp);

  return 0;
}

RandomTestRunner RandomTestRunner::make(uint32_t seed)
{
  RandomTestParam param;

  param.verbose = nnfw::misc::EnvVar("VERBOSE").asInt(0);
  param.tolerance = nnfw::misc::EnvVar("TOLERANCE").asInt(1);
  param.tensor_logging = nnfw::misc::EnvVar("TENSOR_LOGGING").asBool(false);
  param.log_path = nnfw::misc::EnvVar("TENSOR_LOGGING").asString("tensor_log.txt");

  return RandomTestRunner{seed, param};
}
