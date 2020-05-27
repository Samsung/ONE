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

#include "PermutationEliminationPass.h"

#include "ir/Operand.h"
#include "ir/operand/LowerInfo.h"
#include "ir/Graph.h"
#include "backend/IConfig.h"
#include "util/logging.h"

namespace onert
{
namespace ir
{
namespace pass
{
void PermutationEliminationPass::callback(const OperandIndex &inp_index, Operand &object)
{
  if (_graph.getInputs().contains(inp_index))
  {
    eliminateInput(inp_index, object);
  }
  else if (_graph.getOutputs().contains(inp_index))
  {
    eliminateOutput(inp_index, object);
  }
}

void PermutationEliminationPass::eliminateInput(const OperandIndex &inp_index, Operand &object)
{
  auto &model_inputs = _graph.getInputs();

  // get uses of the model's given input
  auto uses = object.getUses();

  // input must be used just by permutation
  if (uses.size() != 1)
  {
    return;
  }

  for (auto input_use : uses)
  {
    auto &perm_operation = _graph.operations().at(input_use);
    auto perm_inputs = perm_operation.getInputs();

    auto perm_outputs = perm_operation.getOutputs();

    if (!isPermuteLayerToEliminate(perm_inputs, perm_outputs, true))
    {
      return;
    }

    assert(perm_inputs.at(0) == inp_index);

    VERBOSE(PermutationEliminationPass::EliminateInput) << "remove NHWC_TO_NCHW permutation\n";

    // set model's new input, which was output of permutation
    model_inputs.replace(inp_index, perm_outputs.at(0));

    // remove model's input, which is also input of permutation
    _graph.removeOperand(inp_index);

    // remove permutation operation
    assert(_lowered_graph.op_seqs().containsOperation(input_use));
    auto op_seq_idx = _lowered_graph.op_seqs().getOperation(input_use);
    _lowered_graph.op_seqs().remove(op_seq_idx);
    _graph.operations().remove(input_use);

    VERBOSE(PermutationEliminationPass::EliminateInput)
        << inp_index.value() << " is model's input and is removed. New input is "
        << perm_outputs.at(0).value() << "\n"
        << input_use.value() << " is removed permutation operation\n";
  }
}

void PermutationEliminationPass::eliminateOutput(const OperandIndex &out_index, Operand &object)
{
  auto &model_outputs = _graph.getOutputs();

  // get defs of the model's given output
  auto defs = object.getDef();

  // output must use just permutation
  if (defs.size() != 1)
  {
    return;
  }

  for (auto output_def : defs)
  {
    auto &perm_operation = _graph.operations().at(output_def);
    auto perm_outputs = perm_operation.getOutputs();

    auto perm_inputs = perm_operation.getInputs();
    if (!isPermuteLayerToEliminate(perm_inputs, perm_outputs, false))
    {
      return;
    }

    assert(perm_outputs.at(0) == out_index);

    VERBOSE(PermutationEliminationPass::EliminateOutput) << "remove NCHW_TO_NHWC permutation\n";

    // Update operations' output that is used by permute operand
    for (auto perm_input_index : perm_inputs)
    {
      auto &perm_input_operand = _graph.operands().at(perm_input_index);
      perm_input_operand.removeUse(output_def);
    }

    // set model's new output, which was input of permutation
    model_outputs.replace(out_index, perm_inputs.at(0));

    // remove model's output, which is also output of permutation
    _graph.removeOperand(out_index);

    // remove permutation operation
    assert(_lowered_graph.op_seqs().containsOperation(output_def));
    auto op_seq_idx = _lowered_graph.op_seqs().getOperation(output_def);
    _lowered_graph.op_seqs().remove(op_seq_idx);
    _graph.operations().remove(output_def);

    VERBOSE(PermutationEliminationPass::EliminateOutput)
        << out_index.value() << " is model's output and is removed. New output is "
        << perm_inputs.at(0).value() << "\n"
        << output_def.value() << " is removed permutation operation\n";
  }
}

bool PermutationEliminationPass::isPermuteLayerToEliminate(const OperandIndexSequence &inp_indexes,
                                                           const OperandIndexSequence &out_indexes,
                                                           bool is_for_model_input)
{
  auto input_def_factors = _lowered_graph.getLowerInfo(inp_indexes.at(0))->def_factors();
  auto output_def_factors = _lowered_graph.getLowerInfo(out_indexes.at(0))->def_factors();

  auto input_layout = input_def_factors.getOnlyElement().layout();
  auto output_layout = output_def_factors.getOnlyElement().layout();

  if (input_def_factors.size() != 1 || output_def_factors.size() != 1)
  {
    return false;
  }

  // all operands' factor must be the same
  for (auto index : inp_indexes)
  {
    auto op_factor_set = _lowered_graph.getLowerInfo(index)->def_factors();
    if (op_factor_set.size() != 1 ||
        input_layout != _lowered_graph.getLowerInfo(index)->def_factors().getOnlyElement().layout())
    {
      return false;
    }
  }
  // all operands' factor must be the same
  for (auto index : out_indexes)
  {
    auto op_factor_set = _lowered_graph.getLowerInfo(index)->def_factors();
    if (op_factor_set.size() != 1 ||
        output_layout !=
            _lowered_graph.getLowerInfo(index)->def_factors().getOnlyElement().layout())
    {
      return false;
    }
  }

  if (is_for_model_input)
  {
    // check if this is NHWC_TO_NCHW permutation: must have single input, which is model's input
    return (inp_indexes.size() == 1 && input_layout == Layout::NHWC &&
            output_layout == Layout::NCHW);
  }

  // check if this is NCHW_TO_NHWC permutation: must have single output, which is model's output
  return (out_indexes.size() == 1 && input_layout == Layout::NCHW && output_layout == Layout::NHWC);
}

} // namespace pass
} // namespace ir
} // namespace onert
