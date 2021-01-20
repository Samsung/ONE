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

#include "luci/Pass/RemoveUnnecessaryStridedSlicePass.h"

#include <luci/IR/CircleNodes.h>

namespace
{

/**
 * @brief   Return value in CircleConst.
 * @details Return value in position on CircleConst with int64 format.
 *          Begin must be larger than or equal to 0. Size must be larger
 *          than or equal to -1.
 */
int64_t value_from_circle_const(const luci::CircleConst *node, uint32_t idx)
{
  assert(node->rank() == 1 && node->dim(0).value() > idx);
  assert(node->dtype() == loco::DataType::S64 || node->dtype() == loco::DataType::S32);

  if (node->dtype() == loco::DataType::S64)
    return node->at<loco::DataType::S64>(idx);
  return static_cast<int64_t>(node->at<loco::DataType::S32>(idx));
}

bool remove_no_effect_strided_slice(luci::CircleNode *node)
{
  auto target_node = dynamic_cast<luci::CircleStridedSlice *>(node);
  if (target_node == nullptr)
    return false;

  auto begin_const = dynamic_cast<luci::CircleConst *>(target_node->begin());
  if (begin_const == nullptr)
    return false;

  auto strides_const = dynamic_cast<luci::CircleConst *>(target_node->strides());
  if (strides_const == nullptr)
    return false;

  auto input_node = loco::must_cast<luci::CircleNode *>(target_node->input());
  if (input_node == nullptr)
    return false;

  auto end_const = dynamic_cast<luci::CircleConst *>(target_node->end());
  if (end_const == nullptr)
    return false;

  for (uint32_t i = 0; i < input_node->rank(); i++)
  {
    int64_t begin_value = value_from_circle_const(begin_const, i);
    if (begin_value != 0)
      return false;

    int64_t strides_value = value_from_circle_const(strides_const, i);
    if (strides_value == -1)
      continue;

    if (strides_value != 1)
      return false;

    int64_t end_value = value_from_circle_const(end_const, i);
    if (end_value == -1)
      continue;

    if (end_value != input_node->dim(i).value())
      return false;

    if (!input_node->dim(i).known())
      return false;
  }
  replace(target_node).with(input_node);
  return true;
}

} // namespace

namespace luci
{
/**
 * BEFORE
 *
 *    [CircleNode]
 *          |
 *    [CircleStridedSlice]
 *          |
 *    [CircleNode]
 *
 * AFTER
 *
 *    [CircleNode]
 *          |
 *    [CircleNode]
 *
 * StridedSlice OP has no effect if,
 *    1. Static Shape : begin_const[idx] is 0 AND strides_const[idx] is (-1 OR input_dimension[idx])
 *    2. Dynamic Shape : begin_const[idx] is 0 AND strides_const[idx] is -1
 */
bool RemoveUnnecessaryStridedSlicePass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    if (remove_no_effect_strided_slice(circle_node))
    {
      changed = true;
    }
  }
  return changed;
}

} // namespace luci
