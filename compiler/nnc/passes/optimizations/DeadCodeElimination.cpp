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

#include "passes/optimizations/DeadCodeElimination.h"
#include "mir/Graph.h"

#include <algorithm>

using namespace mir;

nnc::PassData nnc::DeadCodeElimination::run(PassData data)
{
  auto graph = static_cast<Graph *>(data);
  assert(graph);

  std::vector<Operation *> sorted_nodes = getSortedNodes(graph);

  auto remove_if_unused = [graph](Operation *op) {
    if (op->getType() == Operation::Type::input || op->getType() == Operation::Type::output)
      return;

    bool has_no_uses =
      std::all_of(op->getOutputs().cbegin(), op->getOutputs().cend(),
                  [](const Operation::Output &output) { return output.getUses().empty(); });

    if (has_no_uses)
    {
      graph->removeNode(op);
    }
  };

  std::for_each(sorted_nodes.rbegin(), sorted_nodes.rend(), remove_if_unused);

  return graph;
}
