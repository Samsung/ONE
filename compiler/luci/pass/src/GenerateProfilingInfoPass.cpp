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

#include "luci/Pass/GenerateProfilingInfoPass.h"

#include <luci/IR/CircleNodes.h>

namespace
{

bool generate_profiling_info(loco::Graph *g)
{
  // all nodes should be Circle dialect.
  for (auto node : loco::all_nodes(g))
  {
    if (dynamic_cast<luci::CircleNode *>(node))
      return false;
  }

  int32_t idx = 0;
  for (auto node : loco::all_nodes(g))
  {
    if (luci::CircleNode *circle_node = dynamic_cast<luci::CircleNode *>(node))
      circle_node->p_index(idx++);
  }

  return true;
}

} // namespace

namespace luci
{

bool GenerateProfilingInfoPass::run(loco::Graph *g)
{
  bool changed = false;

  if (generate_profiling_info(g))
    changed = true;

  return changed;
}

} // namespace luci
