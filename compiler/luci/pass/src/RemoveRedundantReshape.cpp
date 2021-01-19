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

#include "luci/Pass/RemoveRedundantReshapePass.h"

#include <luci/IR/CircleNodes.h>

namespace
{

bool remove_redundant_reshape(luci::CircleNode *node)
{
  auto target_node = dynamic_cast<luci::CircleReshape *>(node);
  if (target_node == nullptr)
    return false;

  auto pred_node = dynamic_cast<luci::CircleReshape *>(target_node->tensor());
  if (pred_node == nullptr)
    return false;

  auto target_shape_node = dynamic_cast<luci::CircleConst *>(target_node->shape());
  if (target_shape_node == nullptr)
    return false;

  auto pred_shape_node = dynamic_cast<luci::CircleConst *>(pred_node->shape());
  if (pred_shape_node == nullptr)
    return false;

  target_node->tensor(pred_node->tensor());
  return true;
}

} // namespace

namespace luci
{

/**
 * BEFORE
 *
 *      [CircleNode]
 *            |
 *    [CircleReshape_1]
 *            |
 *    [CircleReshape_2]
 *            |
 *      [CircleNode]
 *
 * AFTER
 *
 *      [CircleNode]
 *            |
 *    [CircleReshape_2]
 *            |
 *      [CircleNode]
 **/
bool RemoveRedundantReshapePass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    if (remove_redundant_reshape(circle_node))
    {
      changed = true;
    }
  }
  return changed;
}

} // namespace luci
