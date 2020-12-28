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
namespace compiler
{

/**
 * @brief Class that contains lowering information on graph.
 *        In addition, after lowering, operands in graph will be set to "dynamic"
 *        if the shape of output of an operation cannot be decided at compilation time.
 */
class LoweredGraph
{
public:
  LoweredGraph(const ir::Graph &graph, const compiler::CompilerOptions &options);

  ir::Graph &graph() { return _graph; }
  const ir::Graph &graph() const { return _graph; }
  const ir::LowerInfoMap *getLowerInfo() const { return &_lower_info_map; }
  const ir::operation::LowerInfo *getLowerInfo(const ir::OpSequenceIndex &op_seq_index) const;
  void setLowerInfo(const ir::OpSequenceIndex &op_seq_index,
                    std::unique_ptr<ir::operation::LowerInfo> &&lower_info);
  void removeLowerInfo(const ir::OpSequenceIndex &op_seq_index);
  const ir::operand::LowerInfo *getLowerInfo(const ir::OperandIndex &index) const;
  ir::operand::LowerInfo *getLowerInfo(const ir::OperandIndex &index);
  void setLowerInfo(const ir::OperandIndex &index,
                    std::unique_ptr<ir::operand::LowerInfo> &&lower_info);
  void removeLowerInfo(const ir::OperandIndex &index);
  ir::OpSequences &op_seqs() { return _op_seqs; }
  const ir::OpSequences &op_seqs() const { return _op_seqs; }
  void iterateTopolOpSeqs(
    const std::function<void(const ir::OpSequenceIndex &, const ir::OpSequence &)> &fn) const;
  void
  iterateTopolOpSeqs(const std::function<void(const ir::OpSequenceIndex &, ir::OpSequence &)> &fn);
  std::shared_ptr<ir::OperationIndexMap<int64_t>> indexed_ranks() { return _indexed_ranks; }

private:
  void
  makeOpSequences(ir::OperandIndexMap<std::unique_ptr<ir::operand::LowerInfo>> &operands_lower_info,
                  const compiler::CompilerOptions &options,
                  const compiler::BackendResolver &backend_resolver);

  void manipulateLowerInfo(
    ir::OperandIndexMap<std::unique_ptr<ir::operand::LowerInfo>> &operands_lower_info);
  void dumpLowerInfo();
  bool mergeable(const ir::OpSequenceIndex &op_seq_index, const ir::OperationIndex &node_index,
                 ir::Layout layout, const compiler::BackendResolver &backend_resolver);
  ir::OpSequenceIndex appendFreshSingleOpSequence(const ir::OperationIndex &node_index,
                                                  const ir::Operation &node);
  std::vector<ir::OpSequenceIndex> topolSortOpSeqs() const;

private:
  ir::Graph _graph;
  std::shared_ptr<ir::OperationIndexMap<int64_t>> _indexed_ranks;
  ir::LowerInfoMap _lower_info_map;
  // Pass(for Perm) can accept only graph so that Graph has OpSequences as a member
  ir::OpSequences _op_seqs;
};

} // namespace compiler
} // namespace onert

#endif // __ONERT_IR_LOWERED_GRAPH_H__
