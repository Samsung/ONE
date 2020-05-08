/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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
#include <logo/CheckIfDeadNodeService.h>

#include <loco/IR/Algorithm.h>
#include <loco/IR/CanonicalDialect.h>
#include <loco/IR/CanonicalNode.h>

#include <set>

namespace logo
{

bool RemoveDeadNodeWithQueryPass::run(loco::Graph *g)
{
  // Find dead(= non-active) nodes
  std::set<loco::Node *> candidates;

  for (auto node : loco::all_nodes(g))
  {
    // The node's dialect must have a CheckIfDeadNodeService
    if (auto service = node->dialect()->service<CheckIfDeadNodeService>())
    {
      if (service->isDeadNode(node))
      {
        candidates.insert(node);
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
