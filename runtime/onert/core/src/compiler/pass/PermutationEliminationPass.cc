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
#include "backend/controlflow/Config.h"

#include "util/logging.h"

namespace onert
{
namespace compiler
{
namespace pass
{

void PermutationEliminationPass::callback(const ir::OperationIndex &ind, ir::Operation &node)
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
    auto in_def_factor = _lowered_graph.getLowerInfo(in_operand)->def_factors().getOnlyElement();
    auto out_def_factor = _lowered_graph.getLowerInfo(out_operand)->def_factors().getOnlyElement();

    auto in_config = in_def_factor.backend()->config();
    auto out_config = out_def_factor.backend()->config();

    // FIXME Supporting dynamic tensor does not exactly mean those are portable.
    //       It may need to have another config option for checking if each uses `IPortableTensor`.
    if (!(in_config->supportDynamicTensor() && out_config->supportDynamicTensor()))
      return;
  }

  if (_graph.getOutputs().contains(out_operand))
  {
    // Exceptional case : When the output operand is a model output
    // In this case we keep the output and remove the input

    auto &out_operand_obj = _graph.operands().at(out_operand);
    assert(out_operand_obj.getDef() == _op_ind);
    out_operand_obj.unsetDef();
    _lowered_graph.op_seqs().iterate([&](const ir::OpSequenceIndex &, ir::OpSequence &op_seq) {
      if (!op_seq.getOutputs().contains(in_operand))
        return;

      // Update OpSequence/ir::Operation edges and ir::Operand edges
      op_seq.replaceOutputs(in_operand, out_operand);
      for (auto op : op_seq.operations())
      {
        auto &operation_obj = _graph.operations().at(op);
        if (operation_obj.getOutputs().contains(in_operand))
        {
          operation_obj.replaceOutputs(in_operand, out_operand);
          out_operand_obj.setDef(op);
        }
      }
    });

    // Remove Permute operation, enclosing OpSequence and the operand
    {
      _graph.removeOperand(in_operand);

      auto op_seq_ind = _lowered_graph.op_seqs().getOperation(_op_ind);
      // Assumes enclosing OpSequence contatins just this Permute operation
      assert(_lowered_graph.op_seqs().at(op_seq_ind).size() == 1);
      _lowered_graph.op_seqs().remove(op_seq_ind);
      _graph.operations().remove(_op_ind);
    }

    _lowered_graph.op_seqs().iterate([&](const ir::OpSequenceIndex &, ir::OpSequence &op_seq) {
      if (!op_seq.getInputs().contains(in_operand))
        return;

      op_seq.replaceInputs(in_operand, out_operand);
      for (auto op : op_seq.operations())
      {
        auto &operation_obj = _graph.operations().at(op);
        if (operation_obj.getInputs().contains(in_operand))
        {
          operation_obj.replaceInputs(in_operand, out_operand);
          out_operand_obj.insertUse(op);
        }
      }
    });

    VERBOSE(removePermute) << "Permute Op removed, node index : " << _op_ind << std::endl;
    VERBOSE(removePermute) << "  - Input (removed) ir::Operand : " << in_operand << std::endl;
    VERBOSE(removePermute) << "  - Output(kept)    ir::Operand : " << out_operand << std::endl;
  }
  else
  {
    // Otherwise keep the input and remove the output

    auto &in_operand_obj = _graph.operands().at(in_operand);
    in_operand_obj.removeUse(_op_ind);

    // Make OpSequences(that use the output) use the input
    _lowered_graph.op_seqs().iterate([&](const ir::OpSequenceIndex &, ir::OpSequence &op_seq) {
      if (!op_seq.getInputs().contains(out_operand))
        return;

      op_seq.replaceInputs(out_operand, in_operand);
      for (auto op : op_seq.operations())
      {
        auto &operation_obj = _graph.operations().at(op);
        if (operation_obj.getInputs().contains(out_operand))
        {
          operation_obj.replaceInputs(out_operand, in_operand);
          in_operand_obj.insertUse(op);
        }
      }
    });

    // Remove Permute operation, enclosing OpSequence and the operand
    {
      _graph.removeOperand(out_operand);

      auto op_seq_ind = _lowered_graph.op_seqs().getOperation(_op_ind);
      // Assumes enclosing OpSequence contatins just this Permute operation
      assert(_lowered_graph.op_seqs().at(op_seq_ind).size() == 1);
      _lowered_graph.op_seqs().remove(op_seq_ind);
      _graph.operations().remove(_op_ind);
    }

    VERBOSE(removePermute) << "Permute Op removed, node index : " << _op_ind << std::endl;
    VERBOSE(removePermute) << "  - Input (kept)    ir::Operand : " << in_operand << std::endl;
    VERBOSE(removePermute) << "  - Output(removed) ir::Operand : " << out_operand << std::endl;
  }
}

} // namespace pass
} // namespace compiler
} // namespace onert
