/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "GraphIterator.h"

#include "ir/OperationIndexMap.h"
#include "ir/Graph.h"

namespace neurun
{
namespace ir
{

// Explicit instantiations to have implementation in the source file.

template class DefaultIterator<true>;
template class DefaultIterator<false>;

template class PostDfsIterator<true>;
template class PostDfsIterator<false>;

//
// Graph::DefaultIterator
//

template <bool is_const>
void DefaultIterator<is_const>::iterate(GraphRef graph, const IterFn &fn) const
{
  graph.operations().iterate(
      [&](const OperationIndex &index, NodeRef node) -> void { fn(index, node); });
}

//
// Graph::PostDfsIterator
//

template <bool is_const>
void PostDfsIterator<is_const>::iterate(GraphRef graph, const IterFn &fn) const
{
  assert(!graph.isBuildingPhase()); // Restrict iteration condition

  OperationIndexMap<bool> visited;
  graph.operations().iterate([&](const OperationIndex &index, NodeRef) { visited[index] = false; });

  std::function<void(const OperationIndex &, NodeRef)> dfs_recursive =
      [&](const OperationIndex &index, NodeRef node) -> void {
    if (visited[index])
      return;
    visited[index] = true;

    for (auto output : node.getOutputs())
    {
      const auto &operand = graph.operands().at(output);
      for (const auto &use : operand.getUses().list())
      {
        dfs_recursive(use, graph.operations().at(use));
      }
    }

    fn(index, node);
  };

  graph.operations().iterate(dfs_recursive);

  // All of the operations(nodes) must have been visited.
  assert(std::all_of(visited.begin(), visited.end(),
                     [](const std::pair<const OperationIndex, bool> &v) { return v.second; }));
}

} // namespace ir
} // namespace neurun
