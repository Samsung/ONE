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
    assert(same_shape(first_node, second_node)); // FIX_CALLER_UNLESS

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

} // namespace circle_eval_diff

#undef THROW_UNLESS
