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

#include "luci/Pass/FoldDequantize.h"

#include <luci/IR/CircleNodes.h>

#include <loco/Service/TypeInference.h>

namespace
{

bool is_hybrid_supported_op(loco::Node *node)
{
  if (dynamic_cast<luci::CircleFullyConnected *>(node) != nullptr)
    return true;

  return false;
}

bool is_foldable_const(luci::CircleConst *node)
{
  if (node->dtype() == loco::DataType::S8)
    return true;
  if (node->dtype() == loco::DataType::U8)
    return true;

  return false;
}

bool replace_const_node(loco::Node *node, luci::CircleConst *const_node)
{
  if (auto gather = dynamic_cast<luci::CircleGather *>(node))
  {
    gather->params(const_node);
    return true;
  }
  else
  {
    // TODO Support more ops
    return false;
  }
}

luci::CircleConst *dequantized_const_node(luci::CircleConst *const_node)
{
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

  const int32_t q_dim = const_node->quantparam()->quantized_dimension;
  const int32_t q_dim_value = const_node->dim(q_dim).value();

  int32_t right_count = q_dim_value;
  for (uint32_t i = q_dim + 1; i < const_node->rank(); ++i)
    right_count *= const_node->dim(i).value();

  if (const_node->dtype() == loco::DataType::S8)
  {
    for (uint32_t i = 0; i < const_node->size<loco::DataType::S8>(); ++i)
    {
      uint32_t qd = (i % right_count) / (right_count / q_dim_value);
      if (qd >= const_node->quantparam()->zerop.size())
        qd = 0;

      new_const_node->at<loco::DataType::FLOAT32>(i) =
          (float)(const_node->at<loco::DataType::S8>(i) - const_node->quantparam()->zerop.at(qd)) *
          const_node->quantparam()->scale.at(qd);
    }
  }
  else
  {
    for (uint32_t i = 0; i < const_node->size<loco::DataType::U8>(); ++i)
    {
      uint32_t qd = (i % right_count) / (right_count / q_dim_value);
      if (qd >= const_node->quantparam()->zerop.size())
        qd = 0;

      new_const_node->at<loco::DataType::FLOAT32>(i) =
          (float)((int)const_node->at<loco::DataType::U8>(i) -
                  const_node->quantparam()->zerop.at(qd)) *
          const_node->quantparam()->scale.at(qd);
    }
  }

  return new_const_node;
}

} // namespace

namespace luci
{

bool FoldDequantizePass::run(loco::Graph *g)
{
  bool changed = false;

  for (auto node : loco::all_nodes(g))
  {
    if (auto circle_dequant = dynamic_cast<luci::CircleDequantize *>(node))
    {
      auto input_node = circle_dequant->input();
      if (auto const_input = dynamic_cast<luci::CircleConst *>(input_node))
      {
        // Input of Dequantize is constant
        loco::replace(circle_dequant).with(dequantized_const_node(const_input));
        changed = true;
      }
      else if (loco::dtype_get(input_node) == loco::DataType::FLOAT32)
      {
        // Input is already dequantized
        loco::replace(circle_dequant).with(input_node);
        changed = true;
      }
    }
    else if (auto const_node = dynamic_cast<luci::CircleConst *>(node))
    {
      if (const_node->quantparam() != nullptr && is_foldable_const(const_node))
      {
        for (auto s : loco::succs(const_node))
        {
          if (!is_hybrid_supported_op(s))
          {
            // Input can be pre-dequantized
            changed |= replace_const_node(s, dequantized_const_node(const_node));
          }
        }
      }
    }
  }

  return changed;
}

} // namespace luci
