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
#include <ir/Graph.h>
#include <util/Utils.h>

namespace onert
{
namespace compiler
{
namespace pass
{

void ConstantInsertionPass::callback(const ir::OperationIndex &node_index, ir::Operation &node)
{
  const auto op_lower_info = _lowered_graph.getLowerInfo(node_index);
  const auto backend = op_lower_info->backend();
  const auto layout = op_lower_info->layout();
  const auto factor = PermuteFactor{backend, layout};

  for (const auto input : node.getInputs() | ir::Remove::DUPLICATED | ir::Remove::UNDEFINED)
  {
    auto &object = _graph.operands().at(input);

    if (object.isConstant())
    {
      const auto key = ReplaceKey{input, factor};
      if (_replace_operands_map.count(key) == 0)
      {
        ir::Operand new_object(object);
        new_object.unsetDef();
        // TODO Remove const_case
        const_cast<ir::OperationIndexSet &>(new_object.getUses()).clear();
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

      // Remove this node from uses of origin operand
      // Constant operand has no def.
      assert(!object.getDef().valid());
      object.removeUse(node_index);

      // Remove origin operand
      if (object.getUses().size() == 0)
        _graph.removeOperand(input);
    }
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
