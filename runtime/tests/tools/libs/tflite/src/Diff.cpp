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

#include "misc/fp32.h"
#include "misc/tensor/Comparator.h"
#include "misc/tensor/IndexFormatter.h"
#include "misc/tensor/Zipper.h"

#include <tensorflow/lite/c/c_api.h>

#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <stdexcept>

namespace
{

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

} // namespace

namespace nnfw
{
namespace tflite
{

template <typename T>
bool TfLiteInterpMatchApp::compareSingleTensorView(const TensorView<T> &expected,
                                                   const TensorView<T> &obtained, int id) const
{
  std::vector<misc::tensor::Diff<T>> diffs;
  assert(expected.shape() == obtained.shape());

  using nnfw::misc::tensor::Index;
  using nnfw::misc::tensor::zip;

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
bool TfLiteInterpMatchApp::compareSingleTensorView<float>(const TensorView<float> &expected,
                                                          const TensorView<float> &obtained,
                                                          int id) const
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

bool TfLiteInterpMatchApp::run(TfLiteInterpreter &expected, TfLiteInterpreter &obtained) const
{
  auto output_count = TfLiteInterpreterGetOutputTensorCount(&expected);
  assert(output_count == TfLiteInterpreterGetOutputTensorCount(&obtained));

  bool all_matched = true;

  using Comparator = std::function<bool(int32_t, const TfLiteTensor *, const TfLiteTensor *)>;

  std::map<TfLiteType, Comparator> comparators;

  comparators[kTfLiteUInt8] = [this](int32_t id, const TfLiteTensor *expected_tensor,
                                     const TfLiteTensor *obtained_tensor) {
    const auto expected_view = TensorView<uint8_t>::make(expected_tensor);
    const auto obtained_view = TensorView<uint8_t>::make(obtained_tensor);

    return compareSingleTensorView(expected_view, obtained_view, id);
  };

  comparators[kTfLiteInt32] = [this](int32_t id, const TfLiteTensor *expected_tensor,
                                     const TfLiteTensor *obtained_tensor) {
    const auto expected_view = TensorView<int32_t>::make(expected_tensor);
    const auto obtained_view = TensorView<int32_t>::make(obtained_tensor);

    return compareSingleTensorView(expected_view, obtained_view, id);
  };

  comparators[kTfLiteFloat32] = [this](int32_t id, const TfLiteTensor *expected_tensor,
                                       const TfLiteTensor *obtained_tensor) {
    const auto expected_view = TensorView<float>::make(expected_tensor);
    const auto obtained_view = TensorView<float>::make(obtained_tensor);

    return compareSingleTensorView(expected_view, obtained_view, id);
  };

  comparators[kTfLiteBool] = [this](int32_t id, const TfLiteTensor *expected_tensor,
                                    const TfLiteTensor *obtained_tensor) {
    const auto expected_view = TensorView<bool>::make(expected_tensor);
    const auto obtained_view = TensorView<bool>::make(obtained_tensor);

    return compareSingleTensorView(expected_view, obtained_view, id);
  };

  for (int32_t idx = 0; idx < output_count; idx++)
  {
    auto const expected_tensor = TfLiteInterpreterGetOutputTensor(&expected, idx);
    auto const obtained_tensor = TfLiteInterpreterGetOutputTensor(&obtained, idx);
    auto const tensor_type = TfLiteTensorType(expected_tensor);
    assert(tensor_type == TfLiteTensorType(obtained_tensor));

    auto it = comparators.find(tensor_type);

    if (it == comparators.end())
    {
      throw std::runtime_error{"Not supported output type"};
    }

    const auto &comparator = it->second;

    if (!comparator(idx, expected_tensor, obtained_tensor))
    {
      all_matched = false;
    }
  }

  return all_matched;
}

} // namespace tflite
} // namespace nnfw
