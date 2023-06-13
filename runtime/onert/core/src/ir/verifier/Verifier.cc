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

namespace onert
{
namespace ir
{
namespace verifier
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

    for (auto output : node.getOutputs() | Remove::DUPLICATED | Remove::UNDEFINED)
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

//
// EdgeConsistencyVerifier
//

bool EdgeChecker::verify(const Graph &graph) const noexcept
{
  auto &operations = graph.operations();
  uint32_t errors = 0;
  operations.iterate([&](const OperationIndex &index, const IOperation &node) {
    for (auto operand_index : node.getInputs() | ir::Remove::UNDEFINED)
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
    for (auto operand_index : node.getOutputs() | ir::Remove::UNDEFINED)
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
  for (auto operand_ind :
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

} // namespace verifier
} // namespace ir
} // namespace onert
