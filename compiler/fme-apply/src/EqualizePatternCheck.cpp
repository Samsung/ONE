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

#include "EqualizePattern.h"

#include <loco.h>
#include <luci/IR/CircleNode.h>

#include <vector>
#include <map>
#include <string>

using namespace fme_apply;

namespace fme_apply
{

/**
 * It checks if given patterns are valid as follows.
 *
 * - "scale" is empty.
 * - "front" and "back" of the patterns are in the graph.
 */
void check_patterns_valid(loco::Graph *g, const std::vector<EqualizePattern> &patterns)
{
  // Create a map to find node by its name
  std::map<std::string, const luci::CircleNode *> node_by_name;
  {
    for (auto node : loco::active_nodes(loco::output_nodes(g)))
    {
      auto cnode = loco::must_cast<luci::CircleNode *>(node);
      node_by_name[cnode->name()] = cnode;
    }
  }

  for (const auto &p : patterns)
  {
    // "scale" is empty.
    // "scale" is calculated in the runtime.
    if (not p.scale.empty())
    {
      throw std::runtime_error{"'scale' shouldn't exist."};
    }

    // "front" and "back" of the patterns are in the graph.
    if (node_by_name.find(p.front) == node_by_name.end() or
        node_by_name.find(p.back) == node_by_name.end())
    {
      throw std::runtime_error{"Given front or back don't exist in the graph."};
    }
  }
}

} // namespace fme_apply
