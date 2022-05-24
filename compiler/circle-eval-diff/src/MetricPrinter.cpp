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

#include "MetricPrinter.h"

#include <luci/IR/CircleNode.h>

#include <iostream>
#include <cassert>

using Tensor = circle_eval_diff::Tensor;

#define THROW_UNLESS(COND, MSG) \
  if (not(COND))                \
    throw std::runtime_error(MSG);

namespace
{

uint32_t num_elems(const luci::CircleNode *node)
{
  uint32_t res = 1;

  for (uint32_t i = 0; i < node->rank(); i++)
    res *= node->dim(i).value();

  return res;
}

template <typename T> bool same_shape(const T a, const T b)
{
  if (a->rank() != b->rank())
    return false;

  for (uint32_t i = 0; i < a->rank(); i++)
  {
    if (not(a->dim(i) == b->dim(i)))
      return false;
  }

  return true;
}

template <typename T> bool same_dtype(const T a, const T b) { return a->dtype() == b->dtype(); }

template <loco::DataType DT> std::shared_ptr<Tensor> to_fp32(const std::shared_ptr<Tensor> &tensor)
{
  assert(tensor->dtype() == DT); // FIX_CALLER_UNLESS

  auto fp32_tensor = std::make_shared<Tensor>();
  {
    fp32_tensor->dtype(loco::DataType::FLOAT32);
    fp32_tensor->rank(tensor->rank());
    for (uint32_t i = 0; i < tensor->rank(); i++)
      fp32_tensor->dim(i) = tensor->dim(i);

    const auto num_elems = tensor->size<DT>();
    fp32_tensor->size<loco::DataType::FLOAT32>(num_elems);
    for (uint32_t i = 0; i < num_elems; i++)
      fp32_tensor->at<loco::DataType::FLOAT32>(i) = static_cast<float>(tensor->at<DT>(i));
  }
  return fp32_tensor;
}

std::shared_ptr<Tensor> fp32(const std::shared_ptr<Tensor> &tensor)
{
  switch (tensor->dtype())
  {
    case loco::DataType::FLOAT32:
      return tensor;
    case loco::DataType::U8:
      return to_fp32<loco::DataType::U8>(tensor);
    case loco::DataType::S16:
      return to_fp32<loco::DataType::S16>(tensor);
    default:
      throw std::runtime_error("Unsupported data type.");
  }
}

} // namespace

namespace circle_eval_diff
{

void MAEPrinter::init(const luci::Module *first, const luci::Module *second)
{
  THROW_UNLESS(first != nullptr, "Invalid module.");
  THROW_UNLESS(second != nullptr, "Invalid module.");

  const auto first_output = loco::output_nodes(first->graph());
  const auto second_output = loco::output_nodes(second->graph());

  assert(first_output.size() == second_output.size()); // FIX_CALLER_UNLESS

  for (uint32_t i = 0; i < first_output.size(); i++)
  {
    const auto first_node = loco::must_cast<luci::CircleNode *>(first_output[i]);
    const auto second_node = loco::must_cast<luci::CircleNode *>(second_output[i]);

    // Create tensors to store intermediate results
    _intermediate.emplace_back();
    _intermediate.at(i).dtype(loco::DataType::FLOAT32);
    // NOTE Use both first_node and second_node to avoid release build break
    _intermediate.at(i).rank(first_node->rank());
    uint32_t num_elems = 1;
    for (uint32_t j = 0; j < second_node->rank(); j++)
    {
      _intermediate.at(i).dim(j) = second_node->dim(j);
      num_elems *= second_node->dim(j).value();
    }
    _intermediate.at(i).size<loco::DataType::FLOAT32>(num_elems);

    // Check the buffer is initilized with zero
    for (uint32_t j = 0; j < num_elems; j++)
      assert(_intermediate.at(i).at<loco::DataType::FLOAT32>(j) == 0.0);

    // Save output names for logging
    _output_names.emplace_back(first_node->name());
  }
}

void MAEPrinter::accum_absolute_error(uint32_t output_idx, const std::shared_ptr<Tensor> &a,
                                      const std::shared_ptr<Tensor> &b)
{
  assert(a->dtype() == loco::DataType::FLOAT32 and
         b->dtype() == loco::DataType::FLOAT32); // FIX_CALLER_UNLESS
  assert(same_shape(a.get(), b.get()));          // FIX_CALLER_UNLESS
  assert(output_idx < _intermediate.size());     // FIX_CALLER_UNLESS

  for (uint32_t i = 0; i < a->size<loco::DataType::FLOAT32>(); i++)
  {
    _intermediate.at(output_idx).at<loco::DataType::FLOAT32>(i) +=
      std::abs(a->at<loco::DataType::FLOAT32>(i) - b->at<loco::DataType::FLOAT32>(i));
  }
}

void MAEPrinter::accumulate(const std::vector<std::shared_ptr<Tensor>> &first,
                            const std::vector<std::shared_ptr<Tensor>> &second)
{
  assert(first.size() == second.size());        // FIX_CALLER_UNLESS
  assert(first.size() == _intermediate.size()); // FIX_CALLER_UNLESS

  for (uint32_t output_idx = 0; output_idx < _intermediate.size(); output_idx++)
  {
    const auto first_output = first[output_idx];
    const auto second_output = second[output_idx];

    // Cast data to fp32 and then compute absolute error
    const auto fp32_first_output = fp32(first_output);
    const auto fp32_second_output = fp32(second_output);

    accum_absolute_error(output_idx, fp32_first_output, fp32_second_output);
  }

  _num_data++;
}

void MAEPrinter::dump(std::ostream &os) const
{
  os << "Mean Absolute Error (MAE)" << std::endl;

  for (uint32_t output_idx = 0; output_idx < _intermediate.size(); output_idx++)
  {
    const auto name = _output_names.at(output_idx);
    const auto &inter = _intermediate.at(output_idx);
    assert(inter.dtype() == loco::DataType::FLOAT32); // FIX_ME_UNLESS
    const auto elem_count = inter.size<loco::DataType::FLOAT32>();

    // Compute MAE
    float mae = 0.0;
    for (uint32_t elem_idx = 0; elem_idx < elem_count; elem_idx++)
      mae += inter.at<loco::DataType::FLOAT32>(elem_idx);

    mae = mae / elem_count;
    mae = mae / _num_data;

    os << "MAE for " << name << " is " << mae << std::endl;
  }
}

// TODO Remove duplicate codes with MAEPrinter
void MAPEPrinter::init(const luci::Module *first, const luci::Module *second)
{
  THROW_UNLESS(first != nullptr, "Invalid module.");
  THROW_UNLESS(second != nullptr, "Invalid module.");

  const auto first_output = loco::output_nodes(first->graph());
  const auto second_output = loco::output_nodes(second->graph());

  assert(first_output.size() == second_output.size()); // FIX_CALLER_UNLESS

  for (uint32_t i = 0; i < first_output.size(); i++)
  {
    const auto first_node = loco::must_cast<luci::CircleNode *>(first_output[i]);
    const auto second_node = loco::must_cast<luci::CircleNode *>(second_output[i]);

    // Create tensors to store intermediate results
    _intermediate.emplace_back();
    _intermediate.at(i).dtype(loco::DataType::FLOAT32);
    // NOTE Use both first_node and second_node to avoid release build break
    _intermediate.at(i).rank(first_node->rank());
    uint32_t num_elems = 1;
    for (uint32_t j = 0; j < second_node->rank(); j++)
    {
      _intermediate.at(i).dim(j) = second_node->dim(j);
      num_elems *= second_node->dim(j).value();
    }
    _intermediate.at(i).size<loco::DataType::FLOAT32>(num_elems);

    // Check the buffer is initilized with zero
    for (uint32_t j = 0; j < num_elems; j++)
      assert(_intermediate.at(i).at<loco::DataType::FLOAT32>(j) == 0.0);

    // Save output names for logging
    _output_names.emplace_back(first_node->name());
  }
}

// Accumulate |(a - b) / a|
void MAPEPrinter::accum_mean_absolute_error(uint32_t output_idx, const std::shared_ptr<Tensor> &a,
                                            const std::shared_ptr<Tensor> &b)
{
  assert(a->dtype() == loco::DataType::FLOAT32 and
         b->dtype() == loco::DataType::FLOAT32); // FIX_CALLER_UNLESS
  assert(same_shape(a.get(), b.get()));          // FIX_CALLER_UNLESS
  assert(output_idx < _intermediate.size());     // FIX_CALLER_UNLESS

  for (uint32_t i = 0; i < a->size<loco::DataType::FLOAT32>(); i++)
  {
    const auto a_val = a->at<loco::DataType::FLOAT32>(i);
    const auto b_val = b->at<loco::DataType::FLOAT32>(i);
    _intermediate.at(output_idx).at<loco::DataType::FLOAT32>(i) +=
      std::abs((a_val - b_val) / a_val);
  }
}

// Assumption
// first: the result of fp32 model
// second: the result of fake-quantized model
void MAPEPrinter::accumulate(const std::vector<std::shared_ptr<Tensor>> &first,
                             const std::vector<std::shared_ptr<Tensor>> &second)
{
  assert(first.size() == second.size());        // FIX_CALLER_UNLESS
  assert(first.size() == _intermediate.size()); // FIX_CALLER_UNLESS

  for (uint32_t output_idx = 0; output_idx < _intermediate.size(); output_idx++)
  {
    const auto first_output = first[output_idx];
    const auto second_output = second[output_idx];

    // Cast data to fp32 and then compute absolute error
    const auto fp32_first_output = fp32(first_output);
    const auto fp32_second_output = fp32(second_output);

    accum_mean_absolute_error(output_idx, fp32_first_output, fp32_second_output);
  }

  _num_data++;
}

void MAPEPrinter::dump(std::ostream &os) const
{
  os << "Mean Absolute Percentage Error (MAPE)" << std::endl;

  for (uint32_t output_idx = 0; output_idx < _intermediate.size(); output_idx++)
  {
    const auto name = _output_names.at(output_idx);
    const auto &inter = _intermediate.at(output_idx);
    assert(inter.dtype() == loco::DataType::FLOAT32); // FIX_ME_UNLESS
    const auto elem_count = inter.size<loco::DataType::FLOAT32>();

    // Compute MAPE
    float mape = 0.0;
    for (uint32_t elem_idx = 0; elem_idx < elem_count; elem_idx++)
      mape += inter.at<loco::DataType::FLOAT32>(elem_idx);

    mape = mape / elem_count;
    mape = mape / _num_data;
    mape *= 100.0;

    os << "MAPE for " << name << " is " << mape << "%" << std::endl;
  }
}

// TODO Remove duplicate codes with MAEPrinter
void MPEIRPrinter::init(const luci::Module *first, const luci::Module *second)
{
  THROW_UNLESS(first != nullptr, "Invalid module.");
  THROW_UNLESS(second != nullptr, "Invalid module.");

  const auto first_output = loco::output_nodes(first->graph());
  const auto second_output = loco::output_nodes(second->graph());

  assert(first_output.size() == second_output.size()); // FIX_CALLER_UNLESS

  for (uint32_t i = 0; i < first_output.size(); i++)
  {
    const auto first_node = loco::must_cast<luci::CircleOutput *>(first_output[i]);
    const auto second_node = loco::must_cast<luci::CircleOutput *>(second_output[i]);

    // Create places to store intermediate results
    _intermediate.emplace_back(0.0);

    // Save output names for logging
    _output_names.emplace_back(first_node->name());
  }
}

// Accumulate PEIR (Peak Error to Interval Ratio)
// PEIR = max(|a - b|) / (max(a) - min(a))
// PEIR >= 0 (lower is better)
void MPEIRPrinter::accum_peir(uint32_t output_idx, const std::shared_ptr<Tensor> &a,
                              const std::shared_ptr<Tensor> &b)
{
  assert(a->dtype() == loco::DataType::FLOAT32 and
         b->dtype() == loco::DataType::FLOAT32); // FIX_CALLER_UNLESS
  assert(same_shape(a.get(), b.get()));          // FIX_CALLER_UNLESS
  assert(output_idx < _intermediate.size());     // FIX_CALLER_UNLESS

  float min = std::numeric_limits<float>::max();
  float max = std::numeric_limits<float>::lowest();

  for (uint32_t i = 0; i < a->size<loco::DataType::FLOAT32>(); i++)
  {
    const auto a_val = a->at<loco::DataType::FLOAT32>(i);
    min = std::min(a_val, min);
    max = std::max(a_val, max);
  }

  float interval = max - min;

  // Corner case: All values are the same. We set interval = 1 in this case
  if (interval == 0)
    interval = 1.0;

  float peak_error = std::numeric_limits<float>::lowest();

  for (uint32_t i = 0; i < a->size<loco::DataType::FLOAT32>(); i++)
  {
    const auto a_val = a->at<loco::DataType::FLOAT32>(i);
    const auto b_val = b->at<loco::DataType::FLOAT32>(i);
    const auto error = std::abs(a_val - b_val);
    peak_error = std::max(error, peak_error);
  }

  _intermediate.at(output_idx) += peak_error / interval;
}

// Assumption (when testing the accuracy of quantized model)
// first: the result of fp32 model
// second: the result of fake-quantized model
void MPEIRPrinter::accumulate(const std::vector<std::shared_ptr<Tensor>> &first,
                              const std::vector<std::shared_ptr<Tensor>> &second)
{
  assert(first.size() == second.size());        // FIX_CALLER_UNLESS
  assert(first.size() == _intermediate.size()); // FIX_CALLER_UNLESS

  for (uint32_t output_idx = 0; output_idx < _intermediate.size(); output_idx++)
  {
    const auto first_output = first[output_idx];
    const auto second_output = second[output_idx];

    // Cast data to fp32 for ease of computation
    const auto fp32_first_output = fp32(first_output);
    const auto fp32_second_output = fp32(second_output);

    accum_peir(output_idx, fp32_first_output, fp32_second_output);
  }

  _num_data++;
}

void MPEIRPrinter::dump(std::ostream &os) const
{
  os << "Mean Peak Error to Interval Ratio (MPEIR)" << std::endl;

  for (uint32_t output_idx = 0; output_idx < _intermediate.size(); output_idx++)
  {
    const auto name = _output_names.at(output_idx);
    const auto sum_of_peir = _intermediate.at(output_idx);

    // Compute MPEIR
    float mpeir = sum_of_peir / _num_data;

    os << "MPEIR for " << name << " is " << mpeir << std::endl;
  }
}

// TODO Remove duplicate codes with MAEPrinter
void TopKMatchPrinter::init(const luci::Module *first, const luci::Module *second)
{
  THROW_UNLESS(first != nullptr, "Invalid module.");
  THROW_UNLESS(second != nullptr, "Invalid module.");

  const auto first_output = loco::output_nodes(first->graph());
  const auto second_output = loco::output_nodes(second->graph());

  assert(first_output.size() == second_output.size()); // FIX_CALLER_UNLESS

  for (uint32_t i = 0; i < first_output.size(); i++)
  {
    const auto first_node = loco::must_cast<luci::CircleOutput *>(first_output[i]);
    const auto second_node = loco::must_cast<luci::CircleOutput *>(second_output[i]);

    // Create places to store intermediate results
    _intermediate.emplace_back(0.0);

    // Save output names for logging
    _output_names.emplace_back(first_node->name());

    // If num_elems of an output is less than k,
    // the output index is added to the skip list
    if (num_elems(first_node) < _k)
    {
      std::cout << "Top-" << _k << "metric for " << first_node->name()
                << " is ignored, because it has elements less than " << _k << std::endl;
      _skip_output.emplace_back(i);
    }
  }
}

void TopKMatchPrinter::accum_topk_accuracy(uint32_t output_idx, const std::shared_ptr<Tensor> &a,
                                           const std::shared_ptr<Tensor> &b)
{
  assert(a->dtype() == loco::DataType::FLOAT32 and
         b->dtype() == loco::DataType::FLOAT32); // FIX_CALLER_UNLESS
  assert(same_shape(a.get(), b.get()));          // FIX_CALLER_UNLESS
  assert(output_idx < _intermediate.size());     // FIX_CALLER_UNLESS

  // Find Top-k largest elements
  // This implementation is a variant of "Method 2 (Use temporary array)" in
  // https://www.geeksforgeeks.org/k-largestor-smallest-elements-in-an-array/
  // We sort top-k elements by value and index to ensure that the element with an earlier
  // index comes first if multiple elements have the same value.
  auto find_topk = [this](const std::shared_ptr<Tensor> &tensor) {
    assert(_k <= tensor->size<loco::DataType::FLOAT32>()); // FIX_CALLER_UNLESS

    // first: value, second: index
    std::vector<std::pair<float, uint32_t>> topk;
    topk.resize(_k);

    // Initialize
    for (uint32_t i = 0; i < _k; i++)
    {
      topk[i] = std::make_pair(tensor->at<loco::DataType::FLOAT32>(i), i);
    }

    // Input pair: (value, index)
    // Return true if a has smaller value than b. If a and b have the same value,
    // return true if a has larger index.
    auto compare = [](const std::pair<float, uint32_t> &a, const std::pair<float, uint32_t> &b) {
      if (a.first == b.first)
        return a.second > b.second;

      return a.first < b.first;
    };

    for (uint32_t i = _k; i < tensor->size<loco::DataType::FLOAT32>(); i++)
    {
      auto val = std::make_pair(tensor->at<loco::DataType::FLOAT32>(i), i);

      auto min = std::min_element(topk.begin(), topk.end(), compare);
      if (compare(*min, val))
      {
        // val is larger than min. Replace min with val.
        auto min_index = std::distance(topk.begin(), min);
        topk[min_index] = val;
      }
    }

    return topk;
  };

  auto first_topk = find_topk(a);
  auto second_topk = find_topk(b);

  uint32_t matched = 0;
  for (uint32_t i = 0; i < _k; i++)
  {
    for (uint32_t j = 0; j < _k; j++)
    {
      if (first_topk[i].second == second_topk[j].second)
      {
        matched++;
        break;
      }
    }
  }

  float matched_ratio = static_cast<float>(matched) / _k;

  _intermediate.at(output_idx) += matched_ratio;
}

bool TopKMatchPrinter::in_skip_list(uint32_t output_index) const
{
  for (auto skip : _skip_output)
  {
    if (output_index == skip)
      return true;
  }

  return false;
}

void TopKMatchPrinter::accumulate(const std::vector<std::shared_ptr<Tensor>> &first,
                                  const std::vector<std::shared_ptr<Tensor>> &second)
{
  assert(first.size() == second.size());        // FIX_CALLER_UNLESS
  assert(first.size() == _intermediate.size()); // FIX_CALLER_UNLESS

  for (uint32_t output_idx = 0; output_idx < _intermediate.size(); output_idx++)
  {
    if (in_skip_list(output_idx))
      continue;

    const auto first_output = first[output_idx];
    const auto second_output = second[output_idx];

    // Cast data to fp32 for ease of computation
    const auto fp32_first_output = fp32(first_output);
    const auto fp32_second_output = fp32(second_output);

    accum_topk_accuracy(output_idx, fp32_first_output, fp32_second_output);
  }

  _num_data++;
}

void TopKMatchPrinter::dump(std::ostream &os) const
{
  os << "Ratio of Matched Indices between Top-" << _k << " results of the models" << std::endl;

  for (uint32_t output_idx = 0; output_idx < _intermediate.size(); output_idx++)
  {
    if (in_skip_list(output_idx))
      continue;

    const auto name = _output_names.at(output_idx);
    const auto sum_of_topk_accuracy = _intermediate.at(output_idx);

    // Compute TopKMatch
    float mean_topk = sum_of_topk_accuracy / _num_data;

    os << "Mean Top-" << _k << " match ratio for " << name << " is " << mean_topk << std::endl;
  }
}

} // namespace circle_eval_diff

#undef THROW_UNLESS
