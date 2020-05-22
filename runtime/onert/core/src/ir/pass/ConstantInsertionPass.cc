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
namespace ir
{
namespace pass
{

void ConstantInsertionPass::callback(const OperationIndex &node_index, Operation &node)
{
  const auto &op_sequence_index = _lowered_graph.op_seqs().getOperation(node_index);
  const auto op_seq_lower_info = _lowered_graph.getLowerInfo(op_sequence_index);
  const auto backend = op_seq_lower_info->backend();
  const auto layout = op_seq_lower_info->layout();
  const auto factor = operand::PermuteFactor{backend, layout};

  for (auto it = node.getInputs().mbegin(); it != node.getInputs().mend(); ++it)
  {
    const auto input = *it;
    auto &object = _graph.operands().at(input);

    if (object.isConstant())
    {
      const auto key = ReplaceKey{input, factor};
      if (_replace_operands_map.count(key) == 0)
      {
        auto new_object = object;
        // TODO Remove const_case
        const_cast<std::list<OperationIndex> &>(new_object.getDef().list()).clear();
        const_cast<std::list<OperationIndex> &>(new_object.getUses().list()).clear();
        const auto new_index = _graph.operands().emplace(new_object);
        _replace_operands_map[key] = new_index;
      }

      const auto replaced_input = _replace_operands_map[key];
      // Update op_seq
      if (_lowered_graph.op_seqs().at(op_sequence_index).getInputs().contains(input))
      {
        // If op_seqs has inputs as the same constant operand and the inputs is replaced all at
        // once, there is no problem because this doesn't change using iterator.
        _lowered_graph.op_seqs().at(op_sequence_index).replaceInputs(input, replaced_input);
      }

      // Update current input operand only. Don't update all input node in this operation.
      *it = replaced_input;

      // Update operand
      auto &replaced_object = _graph.operands().at(replaced_input);
      replaced_object.appendUse(node_index);

      // Remove this node from uses of origin operand
      // Constant operand has no def.
      assert(object.getDef().size() == 0);
      object.removeUse(node_index);

      // Remove origin operand
      if (object.getUses().size() == 0)
        _graph.removeOperand(input);
    }
  }

  // Now this runtime does not support the node making output as constant
  for (const auto &output : node.getOutputs())
  {
    UNUSED_RELEASE(output);
    assert(!_graph.operands().at(output).isConstant());
  }
}

} // namespace pass
} // namespace ir
} // namespace onert
