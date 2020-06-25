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

#ifndef __ONERT_IR_LOWERED_GRAPH_H__
#define __ONERT_IR_LOWERED_GRAPH_H__

#include "ir/Graph.h"
#include "ir/LowerInfoMap.h"
#include "ir/OpSequences.h"
#include "compiler/BackendResolver.h"
#include "compiler/Compiler.h"

namespace onert
{
namespace ir
{

/**
 * @brief Class that contains lowering information on graph.
 *        In addition, after lowering, operands in graph will be set to "dynamic"
 *        if the shape of output of an operation cannot be decided at compilation time.
 */
class LoweredGraph
{
public:
  LoweredGraph(const Graph &graph, const compiler::CompilerOptions &options);

  Graph &graph() { return _graph; }
  const Graph &graph() const { return _graph; }
  const LowerInfoMap *getLowerInfo() const { return &_lower_info_map; }
  const operation::LowerInfo *getLowerInfo(const OpSequenceIndex &op_seq_index) const;
  void setLowerInfo(const OpSequenceIndex &op_seq_index,
                    std::unique_ptr<operation::LowerInfo> &&lower_info);
  void removeLowerInfo(const OpSequenceIndex &op_seq_index);
  const operand::LowerInfo *getLowerInfo(const OperandIndex &index) const;
  operand::LowerInfo *getLowerInfo(const OperandIndex &index);
  void setLowerInfo(const OperandIndex &index, std::unique_ptr<operand::LowerInfo> &&lower_info);
  void removeLowerInfo(const OperandIndex &index);
  OpSequences &op_seqs() { return _op_seqs; }
  const OpSequences &op_seqs() const { return _op_seqs; }
  void iterateTopolOpSeqs(
      const std::function<void(const OpSequenceIndex &, const OpSequence &)> &fn) const;
  void iterateTopolOpSeqs(const std::function<void(const OpSequenceIndex &, OpSequence &)> &fn);
  const backend::BackendContexts &backend_contexts() { return _backend_contexts; }
  const backend::BackendContexts &backend_contexts() const { return _backend_contexts; }
  std::shared_ptr<ir::OperationIndexMap<int64_t>> indexed_ranks() { return _indexed_ranks; }

private:
  void makeOpSequences(OperandIndexMap<std::unique_ptr<operand::LowerInfo>> &operands_lower_info,
                       const compiler::CompilerOptions &options,
                       const compiler::BackendResolver &backend_resolver);

  void
  manipulateLowerInfo(OperandIndexMap<std::unique_ptr<operand::LowerInfo>> &operands_lower_info,
                      bool is_primary);
  void dumpLowerInfo();
  bool mergeable(const OpSequenceIndex &op_seq_index, const OperationIndex &node_index,
                 Layout layout, const compiler::BackendResolver &backend_resolver);
  OpSequenceIndex appendFreshSingleOpSequence(const OperationIndex &node_index,
                                              const Operation &node);

private:
  Graph _graph;
  backend::BackendContexts _backend_contexts;
  std::shared_ptr<ir::OperationIndexMap<int64_t>> _indexed_ranks;
  LowerInfoMap _lower_info_map;
  // Pass(for Perm) can accept only graph so that Graph has OpSequences as a member
  OpSequences _op_seqs;
};

} // namespace ir
} // namespace onert

#endif // __ONERT_IR_LOWERED_GRAPH_H__
