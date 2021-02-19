/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "MatchApp.h"

#include <tflite/TensorView.h>
#include <misc/tensor/DiffSummary.h>
#include <misc/tensor/IndexFormatter.h>

#include <iostream>
#include <cassert>

namespace nnfw
{
namespace onert_cmp
{

template <typename T>
bool MatchApp::compareSingleTensorView(const nnfw::misc::tensor::Reader<T> &expected,
                                       const nnfw::misc::tensor::Reader<T> &obtained, int id) const
{
  std::vector<nnfw::misc::tensor::Diff<T>> diffs;
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
bool MatchApp::compareSingleTensorView<float>(const nnfw::misc::tensor::Reader<float> &expected,
                                              const nnfw::misc::tensor::Reader<float> &obtained,
                                              int id) const
{
  nnfw::misc::tensor::DiffSummary summary;

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

bool MatchApp::run(::tflite::Interpreter &tflite, IOManager &manager) const
{
  assert(tflite.outputs().size() == manager.outputs());

  bool all_matched = true;

  using Comparator = std::function<bool(int, ::tflite::Interpreter &, uint32_t, IOManager &)>;

  std::map<TfLiteType, Comparator> comparators;

  comparators[kTfLiteUInt8] = [this](int tfl_id, ::tflite::Interpreter &tflite, uint32_t id,
                                     IOManager &manager) {
    const auto expected = nnfw::tflite::TensorView<uint8_t>::make(tflite, tfl_id);
    const auto obtained = manager.outputView<uint8_t>(id);

    return compareSingleTensorView(expected, obtained, id);
  };

  comparators[kTfLiteInt32] = [this](int tfl_id, ::tflite::Interpreter &tflite, uint32_t id,
                                     IOManager &manager) {
    const auto expected = nnfw::tflite::TensorView<int32_t>::make(tflite, tfl_id);
    const auto obtained = manager.outputView<int32_t>(id);

    return compareSingleTensorView(expected, obtained, id);
  };

  comparators[kTfLiteFloat32] = [this](int tfl_id, ::tflite::Interpreter &tflite, uint32_t id,
                                       IOManager &manager) {
    const auto expected = nnfw::tflite::TensorView<float>::make(tflite, tfl_id);
    const auto obtained = manager.outputView<float>(id);

    return compareSingleTensorView(expected, obtained, id);
  };

  comparators[kTfLiteBool] = [this](int tfl_id, ::tflite::Interpreter &tflite, uint32_t id,
                                    IOManager &manager) {
    const auto expected = nnfw::tflite::TensorView<bool>::make(tflite, tfl_id);
    const auto obtained = manager.outputView<bool>(id);

    return compareSingleTensorView(expected, obtained, id);
  };

  comparators[kTfLiteInt64] = [this](int tfl_id, ::tflite::Interpreter &tflite, uint32_t id,
                                     IOManager &manager) {
    const auto expected = nnfw::tflite::TensorView<int64_t>::make(tflite, tfl_id);
    const auto obtained = manager.outputView<int64_t>(id);

    return compareSingleTensorView(expected, obtained, id);
  };

  for (uint32_t i = 0; i < manager.outputs(); i++)
  {
    auto id = tflite.outputs().at(i);
    auto it = comparators.find(tflite.tensor(id)->type);

    if (it == comparators.end())
    {
      throw std::runtime_error{"Not supported output type"};
    }

    const auto &comparator = it->second;

    if (!comparator(id, tflite, i, manager))
    {
      all_matched = false;
    }
  }

  return all_matched;
}

} // namespace onert_cmp
} // namespace nnfw
