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

#include "luci/Pass/SubstituteTransposeToReshapePass.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Profile/CircleNodeOrigin.h>

namespace
{

/**
 * @brief Convert transpose op in a certain condition to reshape op
 * @details Convert transpose op if it have condition below
 *          1. have a CircleConst perm value.
 *          2. input have an unknown dimension less then 2
 *          3. the order of shape that except dim value 1 remains same on input and output
 *             eg) input shape  = (126, 201, 1, 1) => (126, 201)
 *                 output shape = (1, 126, 1, 201) => (126, 201)
 */
bool substitute_transpose_to_reshape(luci::CircleTranspose *node)
{
  auto perm_const = dynamic_cast<luci::CircleConst *>(node->perm());
  if (perm_const == nullptr)
    return false;

  assert(perm_const->dtype() == loco::DataType::S32);

  auto input_node = loco::must_cast<luci::CircleNode *>(node->a());
  if (perm_const->dim(0).value() != input_node->rank())
    return false;

  // If input have more than 2 unknown dimension, transpose will not be changed.
  int count = 0;
  for (uint32_t i = 0; i < input_node->rank(); i++)
    if (!input_node->dim(i).known())
      count++;
  if (count > 1)
    return false;

  uint32_t idx = 0;
  auto size_items = perm_const->size<loco::DataType::S32>();
  for (uint32_t i = 0; i < size_items; i++)
  {
    assert(perm_const->at<loco::DataType::S32>(i) >= 0 &&
           perm_const->at<loco::DataType::S32>(i) < static_cast<int32_t>(input_node->rank()));
    const auto perm_value = static_cast<uint32_t>(perm_const->at<loco::DataType::S32>(i));
    if (input_node->dim(perm_value).known() && input_node->dim(perm_value).value() == 1)
      continue;
    // To check idx values are increasing
    if (idx > perm_value)
      return false;
    idx = perm_value;
  }

  auto name = node->name();
  assert(name.length() > 0);

  auto new_const_node = node->graph()->nodes()->create<luci::CircleConst>();
  new_const_node->dtype(loco::DataType::S32);
  new_const_node->size<loco::DataType::S32>(size_items);
  new_const_node->shape_status(luci::ShapeStatus::VALID);
  new_const_node->rank(1);
  new_const_node->dim(0).set(size_items);
  for (uint32_t i = 0; i < size_items; i++)
  {
    if (input_node->dim(static_cast<uint32_t>(perm_const->at<loco::DataType::S32>(i))).known())
      new_const_node->at<loco::DataType::S32>(i) = static_cast<int32_t>(
        input_node->dim(static_cast<uint32_t>(perm_const->at<loco::DataType::S32>(i))).value());
    else
      new_const_node->at<loco::DataType::S32>(i) = -1;
  }

  auto new_reshape_node = node->graph()->nodes()->create<luci::CircleReshape>();
  new_reshape_node->tensor(input_node);
  new_reshape_node->shape(new_const_node);
  new_reshape_node->name(name + "/Reshape");
  luci::add_origin(new_reshape_node, luci::get_origin(node));
  new_const_node->name(name + "/Reshape/shape");

  replace(node).with(new_reshape_node);
  return true;
}

} // namespace

namespace luci
{

/**
 * BEFORE
 *
 *     [CircleNode]  [CircleConst]
 *          \             /
 *          [CircleTranspose]
 *                 |
 *            [CircleNode]
 *
 * AFTER
 *
 *     [CircleNode]  [CircleConst]
 *           \             /
 *          [CircleReshape]
 *                 |
 *            [CircleNode]
 *
 */
bool SubstituteTransposeToReshapePass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto circle_node = dynamic_cast<luci::CircleTranspose *>(node))
    {
      if (substitute_transpose_to_reshape(circle_node))
      {
        changed = true;
      }
    }
  }
  return changed;
}

} // namespace luci
