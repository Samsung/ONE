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

#include "PermutationEliminationPass.h"

#include "backend/Backend.h"
#include "util/logging.h"

namespace onert
{
namespace compiler
{
namespace pass
{

void PermutationEliminationPass::callback(const ir::OperationIndex &ind, ir::IOperation &node)
{
  _op_ind = ind;
  node.accept(*this);
};

void PermutationEliminationPass::visit(const ir::operation::Permute &node)
{
  auto in_operand = node.getInputs().at(0);
  auto out_operand = node.getOutputs().at(0);

  // Check if two tensors are both portable if not, we can't eliminate the node
  {
    auto &operand_li_map = _lowered_graph.lower_info().operand;
    auto in_def_factor = operand_li_map.getRawPtr(in_operand)->def_factors().getOnlyElement();
    auto out_def_factor = operand_li_map.getRawPtr(out_operand)->def_factors().getOnlyElement();

    auto in_config = in_def_factor.backend()->config();
    auto out_config = out_def_factor.backend()->config();

    // FIXME Supporting dynamic tensor does not exactly mean those are portable.
    //       It may need to have another config option for checking if each uses `IPortableTensor`.
    if (!(in_config->supportDynamicTensor() && out_config->supportDynamicTensor()))
      return;
  }

  if (_graph.getOutputs().contains(out_operand))
  {
    // If the input is a const, we cannot remove it since we cannot put the constant data in the
    // output buffer during prepare phase.
    auto permute_input = node.getInputs().at(0);
    if (_graph.operands().at(permute_input).isConstant())
      return;
    // If the input is a model input, we cannot remove it since our API lets users to set different
    // buffers for inputs and outputs even though one tensor is both at the same time.
    auto permute_output = node.getOutputs().at(0);
    if (_graph.getInputs().contains(permute_input) && _graph.getOutputs().contains(permute_output))
      return;
    // Likewise, if copying between outputs to outputs, keep it.
    if (_graph.getOutputs().contains(permute_input) && _graph.getOutputs().contains(permute_output))
      return;

    // Exceptional case : When the output operand is a model output
    // In this case we keep the output and remove the input

    auto &out_operand_obj = _graph.operands().at(out_operand);
    assert(out_operand_obj.getDef() == _op_ind);
    out_operand_obj.unsetDef();
    _graph.operations().iterate([&](const ir::OperationIndex &op_ind, ir::IOperation &op) {
      if (!op.getOutputs().contains(in_operand))
        return;
      // Update Operation and Operand edges
      op.replaceOutputs(in_operand, out_operand);
      out_operand_obj.setDef(op_ind);
    });

    // Remove Permute operation and the operand
    {
      _graph.removeOperand(in_operand);
      _graph.operations().remove(_op_ind);
    }

    _graph.operations().iterate([&](const ir::OperationIndex &op_ind, ir::IOperation &op) {
      if (!op.getInputs().contains(in_operand))
        return;
      op.replaceInputs(in_operand, out_operand);
      out_operand_obj.insertUse(op_ind);
    });

    VERBOSE(removePermute) << "Permute Op removed, node index : " << _op_ind << std::endl;
    VERBOSE(removePermute) << "  - Input (removed) Operand : " << in_operand << std::endl;
    VERBOSE(removePermute) << "  - Output(kept)    Operand : " << out_operand << std::endl;
  }
  else
  {
    // Otherwise keep the input and remove the output

    auto &in_operand_obj = _graph.operands().at(in_operand);
    in_operand_obj.removeUse(_op_ind);

    // Make operations(that use the output) use the input
    _graph.operations().iterate([&](const ir::OperationIndex &op_ind, ir::IOperation &op) {
      if (!op.getInputs().contains(out_operand))
        return;
      op.replaceInputs(out_operand, in_operand);
      in_operand_obj.insertUse(op_ind);
    });

    // Remove the Permute operation and out_operand
    {
      _graph.removeOperand(out_operand);
      _graph.operations().remove(_op_ind);
    }

    VERBOSE(removePermute) << "Permute Op removed : " << _op_ind << std::endl;
    VERBOSE(removePermute) << "  - Input (kept)    Operand : " << in_operand << std::endl;
    VERBOSE(removePermute) << "  - Output(removed) Operand : " << out_operand << std::endl;
  }
}

} // namespace pass
} // namespace compiler
} // namespace onert
