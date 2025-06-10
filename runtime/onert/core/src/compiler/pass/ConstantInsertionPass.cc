/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ConstantInsertionPass.h"

#include "ir/Graph.h"
#include "util/logging.h"

namespace onert::compiler::pass
{

void ConstantInsertionPass::callback(const ir::OperationIndex &node_index, ir::IOperation &node)
{
  const auto backend = _lowered_graph.lower_info().operation.at(node_index);

  for (const auto &input : node.getUsedInputSet())
  {
    auto &object = _graph.operands().at(input);

    // Skip if the operand is not constant or not shared constant
    if (!object.isConstant() || object.getUses().size() < 2)
      continue;

    // 1st use of shared constant operand. Keep using original operand without insertion of new one
    // Register original operand into keep_operand map for later reuse on same backend
    if (_keep_operands_map.find(input) == _keep_operands_map.end())
    {
      _keep_operands_map.emplace(input, backend);
      continue;
    }

    // Same PermuteFactor with original operand usage. Keep using original operand
    if (_keep_operands_map.at(input) == backend)
      continue;

    // Different backend with original operand

    // Check operand is already created for current input's PermuteFactor
    // If not, create new operand and register to _replace_operands_map
    if (_replace_operands_map.count(backend) == 0)
    {
      ir::Operand new_object(object);
      new_object.clearDefUse();
      const auto new_index = _graph.operands().emplace(new_object);
      _replace_operands_map[backend] = new_index;
    }

    const auto replaced_input = _replace_operands_map[backend];

    // Update the same inputs of a node at once because inputs of an operation have the same
    // backend
    node.replaceInputs(input, replaced_input);

    // Update operand
    auto &replaced_object = _graph.operands().at(replaced_input);
    replaced_object.insertUse(node_index);

    VERBOSE(ConstInsertPass) << "New operand " << replaced_input << " added(copy of " << input
                             << ") for " << backend->config()->id() << std::endl;
    // Remove this node from uses of origin operand
    // Constant operand has no def.
    assert(!object.getDef().valid());
    object.removeUse(node_index);

    // Remain uses by _keep_operands_map
    assert(object.getUses().size() != 0);
  }

  // Now this runtime does not support the node making output as constant
  for ([[maybe_unused]] const auto &output : node.getUsedOutputSet())
  {
    assert(!_graph.operands().at(output).isConstant());
  }
}

} // namespace onert::compiler::pass
