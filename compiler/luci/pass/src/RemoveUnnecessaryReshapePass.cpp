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

#include "luci/Pass/RemoveUnnecessaryReshapePass.h"

#include <luci/IR/CircleNodes.h>

namespace
{

bool remove_no_effect_reshape(luci::CircleNode *node)
{
  auto target_node = dynamic_cast<luci::CircleReshape *>(node);
  if (target_node == nullptr)
    return false;

  auto new_shape = dynamic_cast<luci::CircleConst *>(target_node->shape());
  if (new_shape == nullptr)
    return false;

  // Compare updated shape and input shape.
  auto input_node = loco::must_cast<luci::CircleNode *>(target_node->tensor());
  if (input_node->rank() != new_shape->dim(0).value())
    return false;
  for (uint32_t i = 0; i < input_node->rank(); i++)
  {
    // If update_shape is -1, don't care
    // TODO check updated shape has value -1 at most one.
    if (new_shape->at<loco::DataType::S32>(i) == -1)
      continue;
    // If input_shape dynamic, can't remove this.
    if (!input_node->dim(i).known())
      return false;
    // If input_shape and updated shape differ, also can't remove.
    if (input_node->dim(i).value() != static_cast<uint32_t>(new_shape->at<loco::DataType::S32>(i)))
      return false;
  }

  replace(target_node).with(input_node);
  return true;
}

} // namespace

namespace luci
{

bool RemoveUnnecessaryReshapePass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    if (remove_no_effect_reshape(circle_node))
    {
      changed = true;
    }
  }
  return changed;
}

} // namespace luci
