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

#include "backend/Backend.h"
#include "ir/Graph.h"
#include "util/Utils.h"
#include "util/logging.h"

namespace onert
{
namespace compiler
{
namespace pass
{

void ConstantInsertionPass::callback(const ir::OperationIndex &node_index, ir::IOperation &node)
{
  const auto op_lower_info = _lowered_graph.lower_info().operation.getRawPtr(node_index);
  const auto backend = op_lower_info->backend();
  const auto layout = op_lower_info->layout();
  const auto factor = PermuteFactor{backend, layout};

  for (const auto &input : node.getInputs() | ir::Remove::DUPLICATED | ir::Remove::UNDEFINED)
  {
    auto &object = _graph.operands().at(input);

    // Skip if the operand is not constant or not shared constant
    if (!object.isConstant() || object.getUses().size() < 2)
      continue;

    // 1st use of shared constant operand. Keep using original operand without insertion of new one
    // Register original operand into keep_operand map for later reuse on same PermuteFactor
    if (_keep_operands_map.find(input) == _keep_operands_map.end())
    {
      _keep_operands_map.emplace(input, factor);
      continue;
    }

    // Same PermuteFactor with original operand usage. Keep using original operand
    if (_keep_operands_map.at(input) == factor)
      continue;

    // Different PermuteFactor with original operand

    // Check operand is already created for current input's PermuteFactor
    // If not, create new operand and register to _replace_operands_map
    const auto key = ReplaceKey{input, factor};
    if (_replace_operands_map.count(key) == 0)
    {
      ir::Operand new_object(object);
      new_object.clearDefUse();
      const auto new_index = _graph.operands().emplace(new_object);
      _replace_operands_map[key] = new_index;
    }

    const auto replaced_input = _replace_operands_map[key];

    // Update the same inputs of a node at once because inputs of an operation have the same
    // PermuteFactor
    node.replaceInputs(input, replaced_input);

    // Update operand
    auto &replaced_object = _graph.operands().at(replaced_input);
    replaced_object.insertUse(node_index);

    VERBOSE(ConstInsertPass) << "New operand " << replaced_input << " added(copy of " << input
                             << ") for " << factor << std::endl;
    // Remove this node from uses of origin operand
    // Constant operand has no def.
    assert(!object.getDef().valid());
    object.removeUse(node_index);

    // Remain uses by _keep_operands_map
    assert(object.getUses().size() != 0);
  }

  // Now this runtime does not support the node making output as constant
  for (const auto &output : node.getOutputs() | ir::Remove::DUPLICATED | ir::Remove::UNDEFINED)
  {
    UNUSED_RELEASE(output);
    assert(!_graph.operands().at(output).isConstant());
  }
}

} // namespace pass
} // namespace compiler
} // namespace onert
