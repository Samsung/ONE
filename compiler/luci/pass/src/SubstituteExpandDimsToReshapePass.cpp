/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/SubstituteExpandDimsToReshapePass.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Profile/CircleNodeOrigin.h>

/**
 * @brief Convert expand_dims op to reshape op
 *        (expand_dims op with const axis ONLY)
 * @example
 *             input.shape = [2,3,4]
 *             expand_dims(input, axis=1)
 *
 *             can be converted to
 *
 *             reshape(input, [2,1,3,4])
 *
 * BEFORE
 *           |
 *      [CircleNode]   [CircleConst]
 *           \               /
 *          [CircleExpandDims]
 *                   |
 *              [CircleNode]
 *                   |
 *
 * AFTER
 *           |
 *      [CircleNode]  [CircleConst]
 *           \              /
 *           [CircleReshape]
 *                   |
 *              [CircleNode]
 *                   |
 */
namespace
{

int32_t unknown_dim_count(luci::CircleNode *node)
{
  int32_t count = 0;

  for (uint32_t i = 0; i < node->rank(); ++i)
    if (!node->dim(i).known())
      ++count;

  return count;
}

/**
 * @brief Return value in position on CircleConst as int32 format.
 */
int32_t value_from_circle_const(const luci::CircleConst *node, uint32_t idx)
{
  // Scalar case: rank 0, only one element in CircleConst
  if (node->rank() == 0)
  {
    if (node->dtype() == loco::DataType::S64)
    {
      assert(node->size<loco::DataType::S64>() == 1); // FIX_ME_UNLESS
      return static_cast<int32_t>(node->at<loco::DataType::S64>(0));
    }
    else if (node->dtype() == loco::DataType::S32)
    {
      assert(node->size<loco::DataType::S32>() == 1); // FIX_ME_UNLESS
      return node->at<loco::DataType::S32>(0);
    }
    else
    {
      throw std::runtime_error("Unsupported dtype");
    }
  }

  assert(node->rank() == 1 && node->dim(0).value() > idx);
  assert(node->dtype() == loco::DataType::S64 || node->dtype() == loco::DataType::S32);

  if (node->dtype() == loco::DataType::S64)
    return static_cast<int32_t>(node->at<loco::DataType::S64>(idx));
  return node->at<loco::DataType::S32>(idx);
}

bool substitute_expand_dims_to_reshape(luci::CircleNode *node)
{
  auto target_node = dynamic_cast<luci::CircleExpandDims *>(node);
  if (target_node == nullptr)
    return false;
  auto input_node = loco::must_cast<luci::CircleNode *>(target_node->input());
  if (input_node->rank() <= 0)
    return false;
  auto axis_node = dynamic_cast<luci::CircleConst *>(target_node->axis());
  if (axis_node == nullptr)
    return false;

  if (axis_node->dtype() != loco::DataType::S64 && axis_node->dtype() != loco::DataType::S32)
  {
    // Abnormal model with unexpected dtype in axis
    return false;
  }

  int32_t axis = value_from_circle_const(axis_node, 0);
  if (axis < 0)
    // WHY ADD +1?
    // As minux index is calculated from the last index,
    // it must expect the new expanded(+1) rank.
    axis = axis + static_cast<int32_t>(input_node->rank()) + 1;

  auto name = node->name();
  assert(name.length() > 0);

  auto graph = target_node->graph();
  auto reshape_node = graph->nodes()->create<luci::CircleReshape>();
  reshape_node->tensor(input_node);
  reshape_node->name(name + "/Reshape");
  luci::add_origin(reshape_node, luci::get_origin(node));

  auto const_node = graph->nodes()->create<luci::CircleConst>();
  const_node->dtype(loco::DataType::S32);
  const_node->size<loco::DataType::S32>(input_node->rank() + 1);
  const_node->shape_status(luci::ShapeStatus::VALID);
  const_node->rank(1);
  const_node->dim(0).set(input_node->rank() + 1);
  luci::add_origin(const_node, luci::get_origin(node));

  for (int32_t i = 0; i < static_cast<int32_t>(input_node->rank()) + 1; i++)
  {
    if (i == axis)
    {
      const_node->at<loco::DataType::S32>(i) = 1;
    }
    else if (i < axis)
    {
      const_node->at<loco::DataType::S32>(i) = input_node->dim(i).value();
    }
    else
    {
      const_node->at<loco::DataType::S32>(i) = input_node->dim(i - 1).value();
    }
  }
  const_node->name(name + "/Reshape/shape");
  reshape_node->shape(const_node);
  replace(target_node).with(reshape_node);
  return true;
}

} // namespace

namespace luci
{

bool SubstituteExpandDimsToReshapePass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    if (unknown_dim_count(circle_node) == 0 && substitute_expand_dims_to_reshape(circle_node))
    {
      changed = true;
    }
  }
  return changed;
}

} // namespace luci
