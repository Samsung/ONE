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

#include <logo/RemoveDeadNodeWithQueryPass.h>
#include <logo/DeadNodeQueryService.h>

#include <loco/IR/Algorithm.h>
#include <loco/IR/CanonicalDialect.h>
#include <loco/IR/CanonicalNode.h>

#include <set>

namespace logo
{

bool RemoveDeadNodeWithQueryPass::run(loco::Graph *g)
{
  // Let's enumerate nodes required to compute output nodes
  auto active_nodes = loco::active_nodes(loco::output_nodes(g));

  // List dead(= non-active) nodes candidates
  std::set<loco::Node *> candidates;

  for (auto node : loco::all_nodes(g))
  {
    if (active_nodes.find(node) == active_nodes.end())
    {
      candidates.insert(node);
    }
  }

  // Find the nodes that should not be dead node in candidates
  for (auto node : candidates)
  {
    if (auto service = node->dialect()->service<DeadNodeQueryService>())
    {
      if (!service->isDeadNode(node))
      {
        candidates.erase(node);
      }
    }
  }

  for (auto node : candidates)
  {
    node->drop();
  }

  for (auto node : candidates)
  {
    g->nodes()->destroy(node);
  }

  return candidates.size() > 0;
}

} // namespace logo
