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
#include "compiler/LoweredGraph.h"

namespace onert
{
namespace ir
{

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

    for (const auto output : node.getOutputs() | Remove::DUPLICATED)
    {
      const auto &operand = graph.operands().at(output);
      for (const auto &use : operand.getUses())
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

template <bool is_const>
void PostDfsIterator<is_const>::iterateOpSeqs(LoweredGraphRef lowered_graph,
                                              const OpSeqIterFn &fn) const
{
  std::unordered_map<OpSequenceIndex, bool> visited;
  lowered_graph.op_seqs().iterate(
      [&](const OpSequenceIndex &index, OpSequenceRef) { visited[index] = false; });

  std::function<void(const OpSequenceIndex &, OpSequenceRef)> dfs_recursive =
      [&](const OpSequenceIndex &index, OpSequenceRef op_seq) -> void {
    if (visited[index])
      return;
    visited[index] = true;

    for (const auto output : op_seq.getOutputs() | Remove::DUPLICATED)
    {
      const auto &operand = lowered_graph.graph().operands().at(output);
      for (const auto &use : operand.getUses())
      {
        const auto use_op_seq_index = lowered_graph.op_seqs().getOperation(use);
        dfs_recursive(use_op_seq_index, lowered_graph.op_seqs().at(use_op_seq_index));
      }
    }

    fn(index, op_seq);
  };

  lowered_graph.op_seqs().iterate(dfs_recursive);

  // All of the operations(nodes) must have been visited.
  assert(std::all_of(visited.begin(), visited.end(),
                     [](const std::pair<const OpSequenceIndex, bool> &v) { return v.second; }));
}

// Explicit instantiations to have implementation in the source file.
// NOTE If these instatiations were in the top of this file, `iterate` is compiled and saved in
//      `GraphIterator.cc.o` but `iterateOpSeqs`. This happens only when cross-building for Android.
//      (Maybe a bug of NDK toolchain(clang)?)

template class DefaultIterator<true>;
template class DefaultIterator<false>;

template class PostDfsIterator<true>;
template class PostDfsIterator<false>;

} // namespace ir
} // namespace onert
