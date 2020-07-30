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
namespace ir
{
namespace pass
{

void PermutationEliminationPass::callback(const OperationIndex &ind, Operation &node)
{
  _op_ind = ind;
  node.accept(*this);
};

void PermutationEliminationPass::visit(const operation::Permute &node)
{
  auto in_operand = node.getInputs().at(0);
  auto out_operand = node.getOutputs().at(0);

  // Check if two tensors are both portable
  // TODO Make this general, this is just a workaround to check two tensors are portable
  {
    auto in_def_factor = _lowered_graph.getLowerInfo(in_operand)->def_factors().getOnlyElement();
    auto out_def_factor = _lowered_graph.getLowerInfo(out_operand)->def_factors().getOnlyElement();

    auto in_backend_id = in_def_factor.backend()->config()->id();
    auto out_backend_id = out_def_factor.backend()->config()->id();

    // TODO Fix this workaround that removes only Permute between cpu and controlflow backend.
    //      This should be general.
    if (!((in_backend_id == backend::controlflow::Config::ID && out_backend_id == "cpu") ||
          (in_backend_id == "cpu" && out_backend_id == backend::controlflow::Config::ID)))
      return;
  }

  if (_graph.getOutputs().contains(out_operand))
  {
    // Exceptional case : When the output operand is a model output
    // In this case we keep the output and remove the input

    auto &out_operand_obj = _graph.operands().at(out_operand);
    out_operand_obj.removeDef(_op_ind);
    _lowered_graph.op_seqs().iterate([&](const ir::OpSequenceIndex &, ir::OpSequence &op_seq) {
      if (!op_seq.getOutputs().contains(in_operand))
        return;

      // Update OpSequence/Operation edges and Operand edges
      op_seq.replaceOutputs(in_operand, out_operand);
      for (auto op : op_seq.operations())
      {
        auto &operation_obj = _graph.operations().at(op);
        if (operation_obj.getOutputs().contains(in_operand))
        {
          operation_obj.replaceOutputs(in_operand, out_operand);
          out_operand_obj.insertDef(op);
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
    VERBOSE(removePermute) << "  - Input (removed) Operand : " << in_operand << std::endl;
    VERBOSE(removePermute) << "  - Output(kept)    Operand : " << out_operand << std::endl;
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
    VERBOSE(removePermute) << "  - Input (kept)    Operand : " << in_operand << std::endl;
    VERBOSE(removePermute) << "  - Output(removed) Operand : " << out_operand << std::endl;
  }
}

} // namespace pass
} // namespace ir
} // namespace onert
