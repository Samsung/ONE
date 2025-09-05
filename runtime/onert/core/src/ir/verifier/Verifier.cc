/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Verifier.h"

#include "ir/Graph.h"
#include "ir/OperationIndexMap.h"
#include "util/logging.h"

namespace
{

using namespace onert::ir;

std::set<train::TrainingOperationIndex>
extractOperations(const train::UseDefChains &training_usedefs)
{
  // Extract TrainingOperations from training_usedefs
  std::set<train::TrainingOperationIndex> operations;
  for (const auto &[output, usedefs] : training_usedefs)
  {
    const auto &defs = usedefs.getTrainingDefs();
    for (const auto &node_index : defs)
      if (node_index.valid() && output.valid())
        operations.insert(node_index);
  }

  return operations;
}

std::unordered_map<train::TrainingOperationIndex, std::vector<train::TrainingOperandIndex>>
extractNodeInputs(const train::UseDefChains &training_usedefs)
{
  // Extract inputs of TrainingOperations from training_usedefs
  std::unordered_map<train::TrainingOperationIndex, std::vector<train::TrainingOperandIndex>>
    node_inputs;
  for (const auto &[input, usedefs] : training_usedefs)
  {
    const auto &uses = usedefs.getTrainingUses();
    for (const auto &node_index : uses)
      if (node_index.valid() && input.valid())
        node_inputs[node_index].emplace_back(input);
  }

  return node_inputs;
}

std::unordered_map<train::TrainingOperationIndex, std::vector<train::TrainingOperandIndex>>
extractNodeOutputs(const train::UseDefChains &training_usedefs)
{
  // Extract outputs of TrainingOperations from training_usedefs
  std::unordered_map<train::TrainingOperationIndex, std::vector<train::TrainingOperandIndex>>
    node_outputs;
  for (const auto &[output, usedefs] : training_usedefs)
  {
    const auto &defs = usedefs.getTrainingDefs();
    for (const auto &node_index : defs)
      if (node_index.valid() && output.valid())
        node_outputs[node_index].emplace_back(output);
  }

  return node_outputs;
}

} // namespace

namespace onert::ir::verifier
{

//
// DAGChecker
//

bool DAGChecker::verify(const Graph &graph) const noexcept
{
  auto &operations = graph.operations();
  bool cyclic = false;

  OperationIndexMap<bool> visited;
  operations.iterate(
    [&](const OperationIndex &index, const IOperation &) { visited[index] = false; });
  OperationIndexMap<bool> on_stack = visited; // Copy from visited

  std::function<void(const OperationIndex &index, const IOperation &)> dfs_recursive =
    [&](const OperationIndex &index, const IOperation &node) -> void {
    if (on_stack[index])
      cyclic = true;
    if (visited[index])
      return;
    visited[index] = true;
    on_stack[index] = true;

    for (auto &&output : node.getOutputs() | Remove::DUPLICATED | Remove::UNDEFINED)
    {
      const auto &operand = graph.operands().at(output);
      for (const auto &use : operand.getUses())
      {
        dfs_recursive(use, graph.operations().at(use));
      }
    }

    on_stack[index] = false;
  };

  operations.iterate(dfs_recursive);

  return !cyclic;
}

// TODO Merge with the above DAGChecker::verify(const Graph &)
bool DAGChecker::verify(const train::UseDefChains &training_usedefs) const noexcept
{
  bool cyclic = false;
  const auto operations = extractOperations(training_usedefs);
  auto outputs_map = extractNodeOutputs(training_usedefs);

  std::unordered_map<train::TrainingOperationIndex, bool> visited;
  for (const auto &node_index : operations)
    visited[node_index] = false;
  auto on_stack = visited; // Copy from visited

  std::function<void(const train::TrainingOperationIndex &index)> dfs_recursive =
    [&](const train::TrainingOperationIndex &index) -> void {
    if (on_stack[index])
      cyclic = true;
    if (visited[index])
      return;
    visited[index] = true;
    on_stack[index] = true;

    auto &node_outputs = outputs_map[index];
    for (const auto &output : node_outputs)
    {
      const auto &uses = training_usedefs.at(output).getTrainingUses();
      for (const auto &use : uses)
      {
        dfs_recursive(use);
      }
    }

    on_stack[index] = false;
  };

  for (const auto &node_index : operations)
    dfs_recursive(node_index);

  return !cyclic;
}

//
// EdgeConsistencyVerifier
//

bool EdgeChecker::verify(const Graph &graph) const noexcept
{
  auto &operations = graph.operations();
  uint32_t errors = 0;
  operations.iterate([&](const OperationIndex &index, const IOperation &node) {
    for (auto &&operand_index : node.getInputs() | ir::Remove::UNDEFINED)
    {
      try
      {
        auto &operand = graph.operands().at(operand_index);
        bool operand_has_use = operand.getUses().contains(index);
        if (!operand_has_use)
        {
          VERBOSE(EdgeChecker) << "[ERROR] EDGE MISMATCH : Missing USE edge - Operand "
                               << operand_index << " to Operation " << index << std::endl;
          errors += 1;
        }
      }
      catch (const std::out_of_range &e)
      {
        VERBOSE(EdgeChecker) << "[ERROR] OPEARAND NOT FOUND : Operation " << index
                             << " has Operand " << operand_index
                             << ", but the operand object is not present in the graph" << std::endl;
        errors += 1;
      }
    }
    for (auto &&operand_index : node.getOutputs() | ir::Remove::UNDEFINED)
    {
      try
      {
        auto &operand = graph.operands().at(operand_index);
        if (operand.getDef() != index)
        {
          VERBOSE(EdgeChecker) << "[ERROR] EDGE MISMATCH : Missing DEF edge - Operand"
                               << operand_index << " to Operation " << index << std::endl;
          errors += 1;
        }
      }
      catch (const std::out_of_range &e)
      {
        VERBOSE(EdgeChecker) << "[ERROR] OPEARAND NOT FOUND : Operation " << index
                             << " has Operand " << operand_index
                             << ", but the operand object is not present in the graph" << std::endl;
        errors += 1;
      }
    }
  });

  VERBOSE(EdgeChecker) << "Total Number of errors : " << errors << std::endl;

  return errors == 0;
}

bool InputOutputChecker::verify(const Graph &graph) const noexcept
{
  for (auto &&operand_ind :
       (graph.getInputs() + graph.getOutputs()) | Remove::DUPLICATED | Remove::UNDEFINED)
  {
    if (!graph.operands().exist(operand_ind))
    {
      VERBOSE(InputOutputChecker) << "Input or Output tensor " << operand_ind << " does not exist.";
      return false;
    }
  }
  return true;
}

// TODO Merge with the above EdgeChecker::verify(const Graph &)
bool EdgeChecker::verify(const train::UseDefChains &training_usedefs) const noexcept
{
  const auto operations = extractOperations(training_usedefs);
  auto inputs_map = extractNodeInputs(training_usedefs);
  auto outputs_map = extractNodeOutputs(training_usedefs);
  uint32_t errors = 0;
  for (const auto &index : operations)
  {
    const auto &node_inputs = inputs_map[index];
    for (const auto &operand_index : node_inputs)
    {
      try
      {
        const auto &uses = training_usedefs.at(operand_index).getTrainingUses();
        bool operand_has_use = (uses.find(index) != uses.end());
        if (!operand_has_use)
        {
          VERBOSE(EdgeChecker) << "[ERROR] EDGE MISMATCH : Missing USE edge - Operand "
                               << operand_index << " to Operation " << index << std::endl;
          errors += 1;
        }
      }
      catch (const std::out_of_range &e)
      {
        VERBOSE(EdgeChecker) << "[ERROR] OPEARAND NOT FOUND : Operation " << index
                             << " has Operand " << operand_index
                             << ", but the operand object is not present in the graph" << std::endl;
        errors += 1;
      }
    }

    const auto &node_outputs = outputs_map[index];
    for (const auto &operand_index : node_outputs)
    {
      try
      {
        const auto &defs = training_usedefs.at(operand_index).getTrainingDefs();
        bool operand_has_def = (defs.find(index) != defs.end());
        if (!operand_has_def)
        {
          VERBOSE(EdgeChecker) << "[ERROR] EDGE MISMATCH : Missing DEF edge - Operand"
                               << operand_index << " to Operation " << index << std::endl;
          errors += 1;
        }
      }
      catch (const std::out_of_range &e)
      {
        VERBOSE(EdgeChecker) << "[ERROR] OPEARAND NOT FOUND : Operation " << index
                             << " has Operand " << operand_index
                             << ", but the operand object is not present in the graph" << std::endl;
        errors += 1;
      }
    }
  }

  VERBOSE(EdgeChecker) << "Total Number of errors : " << errors << std::endl;

  return errors == 0;
}

} // namespace onert::ir::verifier
