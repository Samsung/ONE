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

#include "luci/Pass/RemoveNoEffectSlicePass.h"

#include <luci/IR/CircleNodes.h>

namespace
{

/// @brief Return value in CircleConst.
/// @details Return -2 if CircleConst is nullptr or not valid shape, otherwise return value in
///          position on CircleConst with int64 format.
///          on this pass, begin & size must be large or equal to -1, so -2 is invalid value.
int64_t value_from_circle_const(luci::CircleConst *node, uint32_t idx)
{
  if (node == nullptr || node->rank() != 1 || node->dim(0).value() <= idx)
    return -2;
  if (node->dtype() == loco::DataType::S64)
    return node->at<loco::DataType::S64>(idx);
  else if (node->dtype() == loco::DataType::S32)
    return static_cast<int64_t>(node->at<loco::DataType::S32>(idx));
  return -2;
}

bool remove_no_effect_slice(luci::CircleNode *node)
{
  auto target_node = dynamic_cast<luci::CircleSlice *>(node);
  if (target_node == nullptr)
    return false;
  auto begin_const = dynamic_cast<luci::CircleConst *>(target_node->begin());
  if (begin_const == nullptr)
    return false;
  auto size_const = dynamic_cast<luci::CircleConst *>(target_node->size());
  if (size_const == nullptr)
    return false;
  // Check input output shape.
  auto input_node = loco::must_cast<luci::CircleNode *>(target_node->input());
  for (uint32_t i = 0; i < input_node->rank(); i++)
  {
    int64_t size_value = value_from_circle_const(size_const, i);
    if (value_from_circle_const(begin_const, i) != 0)
      return false;
    if (size_value < 0)
    {
      if (size_value != -1)
        return false;
      size_value = static_cast<int64_t>(input_node->dim(i).value());
    }
    else
    {
      if (input_node->shape_signature().rank() != 0 && input_node->shape_signature().dim(i) == -1)
        return false;
    }
    if (size_value != static_cast<int64_t>(input_node->dim(i).value()))
      return false;
  }
  replace(target_node).with(input_node);
  return true;
}

} // namespace

namespace luci
{
/**
 *   BEFORE
 *      |
 * [CircleNode]
 *      |
 * [CircleSlice]
 *      |
 * [CircleNode](with same shape)
 *      |
 *
 *    AFTER
 *      |
 * [CircleNode] Remove Slice OP
 *      |
 *
 * Slice OP is No Effect if,
 * 1. Static Shape : begin value = 0 and size value = -1 or input dimension
 * 2. Dynamic Shape : begin value = 0 and size value = -1
 */
bool RemoveNoEffectSlicePass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    if (remove_no_effect_slice(circle_node))
    {
      changed = true;
    }
  }
  return changed;
}

} // namespace luci
