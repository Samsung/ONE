/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "SharedMemoryOperands.h"

namespace onert::backend::cpu
{

namespace
{
// To handle cases like Reshape->Reshape->Reshape... chain where the memory is shared.
// In such a case we should re-assign indexes to the first Reshape input.
void reassign_indexes_to_single_sources(
  ir::OperandIndexMap<ir::OperandIndex> &shared_memory_operand_map)
{
  for (auto [shared_ind, source_ind] : shared_memory_operand_map)
  {
    bool other_source_found = false;
    auto it = std::end(shared_memory_operand_map);
    while ((it = shared_memory_operand_map.find(source_ind)) != std::end(shared_memory_operand_map))
    {
      source_ind = shared_memory_operand_map[source_ind];
      other_source_found = true;
    }
    if (other_source_found)
    {
      shared_memory_operand_map[shared_ind] = source_ind;
    }
  }
}

bool is_memory_sharing_allowed(const ir::IGraph &graph, const ir::IOperation &op)
{
  const std::unordered_set<ir::OpCode> ops_with_possible_memory_sharing = {
    ir::OpCode::Reshape, ir::OpCode::ExpandDims, ir::OpCode::Squeeze};

  if (ops_with_possible_memory_sharing.find(op.opcode()) ==
      std::end(ops_with_possible_memory_sharing))
  {
    return false;
  }
  if (graph.operands().at(op.getInputs().at(0)).info().isDynamic())
  {
    return false;
  }
  if (graph.operands().at(op.getOutputs().at(0)).info().isDynamic())
  {
    return false;
  }
  const auto op_input_output = {op.getInputs().at(0), op.getOutputs().at(0)};
  const bool is_model_input_output = std::any_of(
    std::begin(op_input_output), std::end(op_input_output), [&graph](const ir::OperandIndex &ind) {
      return graph.getInputs().contains(ind) || graph.getOutputs().contains(ind);
    });
  return !is_model_input_output;
};

} // namespace

ir::OperandIndexMap<ir::OperandIndex> findSharedMemoryOperandIndexes(const ir::IGraph &graph)
{
  ir::OperandIndexMap<ir::OperandIndex> shared_memory_operand_map;
  graph.operations().iterate([&](const ir::OperationIndex &, const ir::IOperation &op) {
    if (is_memory_sharing_allowed(graph, op))
    {
      assert(op.getInputs().size() == 1 || op.getInputs().size() == 2);
      assert(op.getOutputs().size() == 1);
      shared_memory_operand_map[op.getOutputs().at(0)] = op.getInputs().at(0);
    }
  });
  reassign_indexes_to_single_sources(shared_memory_operand_map);
  return shared_memory_operand_map;
}

} // namespace onert::backend::cpu
