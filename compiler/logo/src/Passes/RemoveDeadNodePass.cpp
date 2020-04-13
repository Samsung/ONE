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

#include <logo/RemoveDeadNodePass.h>

#include <loco/IR/Algorithm.h>
#include <loco/IR/CanonicalDialect.h>
#include <loco/IR/CanonicalNode.h>

#include <set>

namespace logo
{

bool RemoveDeadNodePass::run(loco::Graph *g)
{
  // Let's enumerate nodes required to compute output nodes
  auto active_nodes = loco::active_nodes(loco::output_nodes(g));

  // Find dead(= non-active) nodes
  std::set<loco::Node *> candidates;

  for (auto node : loco::all_nodes(g))
  {
    if (active_nodes.find(node) == active_nodes.end())
    {
      candidates.insert(node);
    }
  }

  // Let's drop the references from each dead node first and then remove these dead nodes
  //
  // Why?
  //
  // Let us consider the following example:
  //    %0 = Pull(...)
  //    %1 = ConstGen(...)
  //    %2 = Forward(input: %1)
  //    %3 = Push(from: %0) <- OUTPUT
  //
  // Forward (%2) is dead as it does not contribute to the final result (%3). However, it
  // refers to another dead node (%1).
  //
  // This example indicates that naive implementation results in dangling references.
  //
  // There are two possible solutions:
  //  1. Destroy nodes in topological order
  //  2. Drop the reference first and then destroy them
  //
  // The current implementation takes the latter approach for the simplicity of implementation.
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
