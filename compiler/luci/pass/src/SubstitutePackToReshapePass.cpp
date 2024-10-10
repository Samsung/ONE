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

#include "luci/Pass/SubstitutePackToReshapePass.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Profile/CircleNodeOrigin.h>

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

bool substitute_pack_to_reshape(luci::CircleNode *node)
{
  auto target_node = dynamic_cast<luci::CirclePack *>(node);
  if (target_node == nullptr)
    return false;
  if (target_node->values_count() != 1)
    return false;
  auto value_node = loco::must_cast<luci::CircleNode *>(target_node->values(0));
  if (value_node->shape_status() != luci::ShapeStatus::VALID)
    return false;
  int32_t axis = target_node->axis();
  if (axis < 0)
    axis = axis + static_cast<int32_t>(value_node->rank()) + 1;

  auto name = node->name();
  assert(name.length() > 0);

  auto graph = target_node->graph();
  auto reshape_node = graph->nodes()->create<luci::CircleReshape>();
  reshape_node->tensor(value_node);
  reshape_node->name(name + "/Reshape");
  luci::add_origin(reshape_node, luci::get_origin(node));

  auto const_node = graph->nodes()->create<luci::CircleConst>();
  const_node->dtype(loco::DataType::S32);
  const_node->size<loco::DataType::S32>(value_node->rank() + 1);
  const_node->shape_status(luci::ShapeStatus::VALID);
  const_node->rank(1);
  const_node->dim(0).set(value_node->rank() + 1);
  luci::add_origin(const_node, luci::get_origin(node));
  for (int32_t i = 0; i < static_cast<int32_t>(value_node->rank()) + 1; i++)
  {
    if (i == axis)
    {
      const_node->at<loco::DataType::S32>(i) = 1;
    }
    else if (i < axis)
    {
      const_node->at<loco::DataType::S32>(i) =
        value_node->dim(i).known() ? value_node->dim(i).value() : -1;
    }
    else
    {
      const_node->at<loco::DataType::S32>(i) =
        value_node->dim(i - 1).known() ? value_node->dim(i - 1).value() : -1;
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

/**
 * BEFORE
 *           |
 *      [CircleNode]
 *           |
 *      [CirclePack]
 *           |
 *      [CircleNode]
 *           |
 *
 * AFTER
 *             |
 *        [CircleNode]  [CircleConst]
 *           |   \             /
 *  [CirclePack] [CircleReshape]
 *                      |
 *                 [CircleNode]
 *                      |
 */
bool SubstitutePackToReshapePass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    if (unknown_dim_count(circle_node) <= 1 && substitute_pack_to_reshape(circle_node))
    {
      changed = true;
    }
  }
  return changed;
}

} // namespace luci
