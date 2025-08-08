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

#include "TrainableConstantInsertionPass.h"

#include "ir/Graph.h"
#include "util/logging.h"

namespace onert::compiler::train::pass
{

void TrainableConstantInsertionPass::callback(const ir::OperationIndex &node_index,
                                              ir::IOperation &node)
{
  for (const auto &input : node.getUsedInputSet())
  {
    auto &object = _graph.operands().at(input);

    // Skip if the operand is not constant or not shared constant
    if (!object.isConstant() || object.getUses().size() < 2)
      continue;

    // Insert new operands for shared constant except for the current node.
    const auto uses(object.getUses());
    for (const auto &use_index : uses)
    {
      if (use_index == node_index)
        continue;

      // NOTE The PermuteFactor(backend) of the current node and the use node may be different.
      //      But there is no problem because both nodes' constant operand will have
      //      only one use node.
      const auto new_index = insertNewOperand(object);
      updateUseDef(input, new_index, use_index);
    }

    // The input of the current node will have one use as the current node
    assert(object.getUses().size() == 1 && object.getUses().contains(node_index));
  }
}

ir::OperandIndex TrainableConstantInsertionPass::insertNewOperand(const ir::Operand &object)
{
  ir::Operand new_object(object);
  new_object.clearDefUse();
  return _graph.operands().emplace(new_object);
}

void TrainableConstantInsertionPass::updateUseDef(const ir::OperandIndex &old_index,
                                                  const ir::OperandIndex &new_index,
                                                  const ir::OperationIndex &node_index)
{
  const auto backend = _lowered_graph.lower_info().operation.at(node_index);

  // Update the same inputs of a node at once because inputs of an operation have the same
  // PermuteFactor
  auto &target_node = _graph.operations().at(node_index);
  target_node.replaceInputs(old_index, new_index);

  // Update the new operand
  auto &new_object = _graph.operands().at(new_index);
  new_object.insertUse(node_index);

  VERBOSE(TrainConstInsertPass) << "New operand " << new_index << " added(copy of " << old_index
                                << ") for " << backend->config()->id() << std::endl;

  // Remove the use node from uses of origin operand
  // Constant operand has no def.
  auto &old_object = _graph.operands().at(old_index);
  assert(!old_object.getDef().valid());
  old_object.removeUse(node_index);
}

} // namespace onert::compiler::train::pass
