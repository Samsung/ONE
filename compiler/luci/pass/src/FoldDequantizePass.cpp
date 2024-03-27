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

#include "luci/Pass/FoldDequantizePass.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Profile/CircleNodeOrigin.h>

#include <fp16.h>

namespace
{

bool is_hybrid_kernel_supported(loco::Node *node)
{
  if (dynamic_cast<luci::CircleFullyConnected *>(node) != nullptr)
    return true;

  return false;
}

bool is_foldable_const(luci::CircleConst *node)
{
  if (node->dtype() == loco::DataType::FLOAT16)
    return true;

  if (node->quantparam() == nullptr)
    return false;

  if (node->dtype() == loco::DataType::S4)
    return true;
  if (node->dtype() == loco::DataType::U4)
    return true;
  if (node->dtype() == loco::DataType::S8)
    return true;
  if (node->dtype() == loco::DataType::U8)
    return true;
  if (node->dtype() == loco::DataType::S16)
    return true;
  if (node->dtype() == loco::DataType::S32)
    return true;
  if (node->dtype() == loco::DataType::S64)
    return true;

  return false;
}

luci::CircleConst *dequantized_const_node(luci::CircleConst *const_node)
{
  auto name = const_node->name();
  assert(name.length() > 0);
  auto g = const_node->graph();
  auto new_const_node = g->nodes()->create<luci::CircleConst>();

  new_const_node->dtype(loco::DataType::FLOAT32);
  new_const_node->rank(const_node->rank());
  uint32_t dim_size = 1;
  for (uint32_t i = 0; i < new_const_node->rank(); ++i)
  {
    new_const_node->dim(i) = const_node->dim(i);
    dim_size *= const_node->dim(i).value();
  }
  new_const_node->size<loco::DataType::FLOAT32>(dim_size);
  new_const_node->shape_status(luci::ShapeStatus::VALID);
  new_const_node->name(name + "_DQ");

  if (const_node->dtype() == loco::DataType::FLOAT16)
  {
    for (uint32_t i = 0; i < new_const_node->size<loco::DataType::FLOAT32>(); ++i)
    {
      auto raw = const_node->at<loco::DataType::FLOAT16>(i);
      new_const_node->at<loco::DataType::FLOAT32>(i) = fp16_ieee_to_fp32_value(raw);
    }
    return new_const_node;
  }

  if (const_node->quantparam() == nullptr)
  {
    throw std::runtime_error("Given constant node has no quantization parameter");
  }

  const int32_t q_dim = const_node->quantparam()->quantized_dimension;
  // For scalar, q_dim_value is 1
  // For non-scalar, q_dim_value is the size of quantized dimension
  const int32_t q_dim_value = const_node->rank() == 0 ? 1 : const_node->dim(q_dim).value();

  int32_t right_count = q_dim_value;
  for (uint32_t i = q_dim + 1; i < const_node->rank(); ++i)
    right_count *= const_node->dim(i).value();

  for (uint32_t i = 0; i < new_const_node->size<loco::DataType::FLOAT32>(); ++i)
  {
    uint32_t qd = (i % right_count) / (right_count / q_dim_value);
    if (qd >= const_node->quantparam()->zerop.size())
      qd = 0;

    switch (const_node->dtype())
    {
      case loco::DataType::U4:
        new_const_node->at<loco::DataType::FLOAT32>(i) =
          static_cast<float>(const_node->at<loco::DataType::U4>(i) -
                             const_node->quantparam()->zerop.at(qd)) *
          const_node->quantparam()->scale.at(qd);
        break;
      case loco::DataType::S4:
        new_const_node->at<loco::DataType::FLOAT32>(i) =
          static_cast<float>(const_node->at<loco::DataType::S4>(i) -
                             const_node->quantparam()->zerop.at(qd)) *
          const_node->quantparam()->scale.at(qd);
        break;
      case loco::DataType::S8:
        new_const_node->at<loco::DataType::FLOAT32>(i) =
          static_cast<float>(const_node->at<loco::DataType::S8>(i) -
                             const_node->quantparam()->zerop.at(qd)) *
          const_node->quantparam()->scale.at(qd);
        break;
      case loco::DataType::S16:
        new_const_node->at<loco::DataType::FLOAT32>(i) =
          static_cast<float>(const_node->at<loco::DataType::S16>(i) -
                             const_node->quantparam()->zerop.at(qd)) *
          const_node->quantparam()->scale.at(qd);
        break;
      case loco::DataType::S32:
        new_const_node->at<loco::DataType::FLOAT32>(i) =
          static_cast<float>(const_node->at<loco::DataType::S32>(i) -
                             const_node->quantparam()->zerop.at(qd)) *
          const_node->quantparam()->scale.at(qd);
        break;
      case loco::DataType::S64:
        new_const_node->at<loco::DataType::FLOAT32>(i) =
          static_cast<float>(const_node->at<loco::DataType::S64>(i) -
                             const_node->quantparam()->zerop.at(qd)) *
          const_node->quantparam()->scale.at(qd);
        break;
      case loco::DataType::U8:
        new_const_node->at<loco::DataType::FLOAT32>(i) =
          static_cast<float>(const_node->at<loco::DataType::U8>(i) -
                             const_node->quantparam()->zerop.at(qd)) *
          const_node->quantparam()->scale.at(qd);
        break;
      default:
        throw std::runtime_error("Not supported dtype for FoldDequantizePass");
    }
  }

  return new_const_node;
}

bool replace_const_node(loco::Node *node, luci::CircleConst *const_node)
{
  if (auto gather = dynamic_cast<luci::CircleGather *>(node))
  {
    gather->params(dequantized_const_node(const_node));
    gather->dtype(loco::DataType::FLOAT32);
    return true;
  }
  else
  {
    // TODO Support more ops
    return false;
  }
}

} // namespace

namespace luci
{

/**
 *
 * Folding pattern 1 - When input of Dequantize is foldable constant
 *
 * [Before]
 *     quantized_const_input ---------- Dequantize ---------- Op ---
 *                             +-- Op1_with_quant_input ---
 *                             +-- Op2_with_quant_input ---
 *
 * [After]
 *   dequantized_const_input -------------------------------- Op ---
 *
 *     quantized_const_input ----- Op1_with_quant_input ---
 *                             +-- Op2_with_quant_input ---
 *
 *
 * Folding pattern 2 - When input of Dequantize uses quantized output value
 *
 * [Before]
 *     quantized_const_input ----- Gather ----- Dequantize --- Op ---
 *                             +-- Op1_with_quant_input ---
 *                             +-- Op2_with_quant_input ---
 *
 * [After]
 *   dequantized_const_input ------Gather -------------------- Op ---
 *
 *     quantized_const_input ----- Op1_with_quant_input ---
 *                             +-- Op2_with_quant_input ---
 *
 *
 */
bool FoldDequantizePass::run(loco::Graph *g)
{
  bool changed = false;

  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto circle_dequant = dynamic_cast<luci::CircleDequantize *>(node))
    {
      if (auto const_input = dynamic_cast<luci::CircleConst *>(circle_dequant->input()))
      {
        // Pattern 1 - When input of Dequantize is foldable constant
        if (is_foldable_const(const_input))
        {
          loco::replace(circle_dequant).with(dequantized_const_node(const_input));
          changed = true;
        }
      }
    }
    else if (auto const_node = dynamic_cast<luci::CircleConst *>(node))
    {
      if (is_foldable_const(const_node))
      {
        for (auto const_node_user : loco::succs(const_node))
        {
          // If user is hybrid kernel supported operation, do not dequantize
          if (is_hybrid_kernel_supported(const_node_user))
            continue;

          auto users = loco::succs(const_node_user);
          if (users.size() > 1)
            continue;

          // Pattern 2 - When input of Dequantize uses quantized output value
          if (auto dequant = dynamic_cast<luci::CircleDequantize *>(*users.begin()))
          {
            if (replace_const_node(const_node_user, const_node))
            {
              loco::replace(dequant).with(const_node_user);
              luci::add_origin(loco::must_cast<luci::CircleNode *>(const_node_user),
                               luci::get_origin(dequant));
              changed = true;
            }
          }
        }
      }
    }
  }

  return changed;
}

} // namespace luci
