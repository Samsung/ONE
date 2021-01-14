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

#include "luci/Pass/RemoveUnnecessarySplitPass.h"

#include <luci/IR/CircleNodes.h>

namespace
{
bool remove_unnecessary_split(luci::CircleNode *node)
{
  auto target_node = dynamic_cast<luci::CircleSplitOut *>(node);
  if (target_node == nullptr)
    return false;

  auto split_node = dynamic_cast<luci::CircleSplit *>(target_node->input());
  if (split_node == nullptr)
    return false;

  if (loco::succs(split_node).size() != 1)
    return false;

  if (split_node->num_split() == 1)
  {
    auto input_node = loco::must_cast<luci::CircleNode *>(split_node->input());
    replace(target_node).with(input_node);
    return true;
  }
  return false;
}

} // namespace

namespace luci
{

bool RemoveUnnecessarySplitPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    if (remove_unnecessary_split(circle_node))
    {
      changed = true;
    }
  }
  return changed;
}

} // namespace luci
