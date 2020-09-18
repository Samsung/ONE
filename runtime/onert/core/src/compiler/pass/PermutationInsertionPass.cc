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

#include "PermutationInsertionPass.h"

#include <cassert>
#include <utility>
#include <unordered_map>

#include "backend/controlflow/Config.h"
#include "ir/Operand.h"
#include "ir/operation/LowerInfo.h"
#include "ir/Graph.h"
#include "backend/IConfig.h"
#include "util/logging.h"
#include <memory>
#include "ir/operation/Permute.h"

namespace onert
{
namespace compiler
{
namespace pass
{

void PermutationInsertionPass::callback(const ir::OperandIndex &index, ir::Operand &object)
{
  auto &&operand_li = _lowered_graph.getLowerInfo(index);
  assert(operand_li);

  // NOTE Later, constants also will have Def
  // Ignore constants
  if (operand_li->def_factors().size() == 0)
  {
    return;
  }

  std::list<ir::OperationIndex> permute_indexes;

  // Build a map for all necessary type of operands
  std::unordered_map<ir::operand::PermuteFactor, ir::OperandIndex> factor_to_index;
  {
    assert(operand_li->def_factors().size() == 1);
    for (auto factor : operand_li->def_factors())
    {
      factor_to_index.emplace(factor, index);
    }

    auto insert_set = operand_li->use_factors() - operand_li->def_factors();
    for (auto factor : insert_set)
    {
      const auto permute_operation_index = insertPermute(index, factor);
      permute_indexes.push_back(permute_operation_index);
      const auto &permute_operation = _graph.operations().at(permute_operation_index);
      const auto permuted_operand_index = permute_operation.getOutputs().at(0);
      factor_to_index.emplace(factor, permuted_operand_index);
    }
  }

  // Update operations' input that uses this operand
  {
    std::list<ir::OperationIndex> remove_list;

    auto uses = object.getUses();
    for (auto use : uses)
    {
      // If permute operation, ignore it
      if (std::find(permute_indexes.begin(), permute_indexes.end(), use) != permute_indexes.end())
        continue;

      auto &operation = _graph.operations().at(use);
      assert(_lowered_graph.op_seqs().containsOperation(use));
      auto op_seq_index = _lowered_graph.op_seqs().getOperation(use);
      auto op_seq_li = _lowered_graph.getLowerInfo(op_seq_index);
      assert(op_seq_li);
      const auto op_seq_layout = op_seq_li->layout();
      const backend::Backend *backend = op_seq_li->backend();
      assert(backend);
      auto use_node_inputs = operation.getInputs();
      assert(use_node_inputs.contains(index));

      auto new_index = factor_to_index.at({backend, op_seq_layout});
      if (index != new_index)
      {
        // Update from op_seq
        // Replace the same inputs of an OpSequence at once for the following reasons:
        // 1. An OpSequence's inputs are the same inputs of first operation
        // 2. An OpSequence may have inputs as the same operand (2 or more).
        // 3. The same inputs of OpSequence have the same PermuteFactor.
        _lowered_graph.op_seqs().at(op_seq_index).replaceInputs(index, new_index);

        // Update from operation
        // Replace the same inputs of an operation at once for the following reasons:
        // No. 2 and 3 above
        operation.replaceInputs(index, new_index);

        // Update from operand
        remove_list.push_back(
            use); // Removal should be done in another loop since we are in the loop
        _graph.operands().at(new_index).insertUse(use);
      }
    }

    for (auto &operation : remove_list)
    {
      object.removeUse(operation);
    }
  }
}

ir::OperationIndex PermutationInsertionPass::insertPermute(const ir::OperandIndex &operand_index,
                                                           const ir::operand::PermuteFactor &factor)
{
  assert(!_graph.isBuildingPhase());

  auto &operand = _graph.operands().at(operand_index);

  // Generate output operand and permute operation
  auto out_operand_index = _graph.addOperand(operand.shape(), operand.typeInfo());
  // change model output if operand_index is model output index and the out operand is controlflow
  // backend
  auto &model_outputs = _graph.getOutputs();
  const backend::Backend *cf_backend = compiler::BackendManager::get().getControlflow();
  if (model_outputs.contains(operand_index) && factor.backend() == cf_backend)
  {
    model_outputs.replace(operand_index, out_operand_index);
  }

  // Find Permute information
  auto input_factor = _lowered_graph.getLowerInfo(operand_index)->def_factors().getOnlyElement();
  auto input_backend = input_factor.backend();
  auto output_backend = factor.backend();
  // NOTE Permute may not have specific layout because the layout of input and output may be
  // different.
  const auto permute_node_layout = ir::Layout::UNKNOWN;
  // NOTE If one backend supports several layout, the backend must support Permute operation
  const backend::Backend *permute_node_backend = compiler::BackendManager::get().getControlflow();
  if (input_backend == output_backend)
  {
    permute_node_backend = input_backend;
  }
  const ir::operand::PermuteFactor permute_node_factor{permute_node_backend, permute_node_layout};

  // Update LowerInfo of input operand
  auto operand_lower_info = _lowered_graph.getLowerInfo(operand_index);
  operand_lower_info->removeUsePermuteFactor(factor);
  operand_lower_info->addUsePermuteFactor(permute_node_factor);

  // Update LowerInfo of output operand
  auto out_operand_li = std::make_unique<ir::operand::LowerInfo>();

  // The input and output factors of all nodes will be the same except Permute. So Tensor's
  // allocators allocates memory using only the information of def permutation factor now.
  // TODO Change param to permute_node_factor
  out_operand_li->addDefPermuteFactor(factor);
  out_operand_li->addUsePermuteFactor(factor);
  _lowered_graph.setLowerInfo(out_operand_index, std::move(out_operand_li));

  // Insert permute operation to the graph
  const auto input_layout = input_factor.layout();
  const auto output_layout = factor.layout();
  using Permute = ir::operation::Permute;
  const auto permute_type = [&]() {
    if (input_layout == ir::Layout::NHWC && output_layout == ir::Layout::NCHW)
    {
      return Permute::Type::NHWC_TO_NCHW;
    }
    else if (input_layout == ir::Layout::NCHW && output_layout == ir::Layout::NHWC)
    {
      return Permute::Type::NCHW_TO_NHWC;
    }
    else
    {
      return Permute::Type::COPY;
    }
  }();
  auto insert_node = std::make_unique<Permute>(operand_index, out_operand_index, permute_type);

  auto node_index = _graph.operations().push(std::move(insert_node));
  const auto &node = _graph.operations().at(node_index);

  VERBOSE_F() << "Permute Op inserted, node index : " << node_index << std::endl;
  VERBOSE_F() << "  - Input (original) Operand : " << operand_index << "("
              << input_factor.backend()->config()->id() << ")" << std::endl;
  VERBOSE_F() << "  - Output(inserted) Operand : " << out_operand_index << "("
              << factor.backend()->config()->id() << ")" << std::endl;

  // OpSequence
  {
    auto op_seq_index = _lowered_graph.op_seqs().emplace(node_index, permute_node_layout);
    auto &op_seq = _lowered_graph.op_seqs().at(op_seq_index);
    op_seq.setInputs(node.getInputs());
    op_seq.setOutputs(node.getOutputs());
    _lowered_graph.setLowerInfo(op_seq_index, std::make_unique<ir::operation::LowerInfo>(
                                                  permute_node_backend, permute_node_layout));
  }

  // Update Use/Def info
  {
    _graph.operands().at(operand_index).insertUse(node_index);
    _graph.operands().at(out_operand_index).setDef(node_index);
  }
  return node_index;
}
} // namespace pass
} // namespace compiler
} // namespace onert
