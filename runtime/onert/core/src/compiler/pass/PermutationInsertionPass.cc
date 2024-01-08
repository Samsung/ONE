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

#include "../../backend/builtin/Config.h"

#include "compiler/OperationLowerInfo.h"
#include "ir/operation/Permute.h"
#include "util/logging.h"

#include <cassert>
#include <memory>
#include <unordered_map>
#include <utility>

namespace onert
{
namespace compiler
{
namespace pass
{

void PermutationInsertionPass::callback(const ir::OperandIndex &index, ir::Operand &object)
{
  auto &operand_li_map = _lowered_graph.lower_info().operand;
  auto &&operand_li = operand_li_map.getRawPtr(index);
  assert(operand_li);

  // NOTE Later, constants also will have Def
  // Ignore constants
  if (operand_li->def_factors().size() == 0)
  {
    return;
  }

  std::list<ir::OperationIndex> permute_indexes;

  // Build a map for all necessary type of operands
  std::unordered_map<PermuteFactor, ir::OperandIndex> factor_to_index;
  {
    assert(operand_li->def_factors().size() == 1);
    for (auto &&factor : operand_li->def_factors())
    {
      factor_to_index.emplace(factor, index);
    }

    auto insert_set = operand_li->use_factors() - operand_li->def_factors();
    for (auto &&factor : insert_set)
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
    for (auto &&use : uses)
    {
      // If permute operation, ignore it
      if (std::find(permute_indexes.begin(), permute_indexes.end(), use) != permute_indexes.end())
        continue;

      auto &operation = _graph.operations().at(use);
      auto op_li = _lowered_graph.lower_info().operation.getRawPtr(use);
      assert(op_li);
      const auto op_layout = op_li->layout();
      const backend::Backend *backend = op_li->backend();
      assert(backend);

      auto new_index = factor_to_index.at({backend, op_layout});
      if (index != new_index)
      {
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

    for (const auto &operation_index : remove_list)
    {
      object.removeUse(operation_index);
    }
  }
}

ir::OperationIndex PermutationInsertionPass::insertPermute(const ir::OperandIndex &operand_index,
                                                           const PermuteFactor &factor)
{
  auto &operand = _graph.operands().at(operand_index);

  // Generate output operand and permute operation
  auto out_operand_index = _graph.addOperand(operand.shape(), operand.typeInfo());
  // change model output if operand_index is model output index and the out operand is builtin
  // backend
  auto &model_outputs = _graph.getOutputs();
  const backend::Backend *builtin_backend = compiler::BackendManager::get().getBuiltin();
  assert(builtin_backend->config()->id() == onert::backend::builtin::Config::ID);

  if (model_outputs.contains(operand_index) && factor.backend() == builtin_backend)
  {
    model_outputs.replace(operand_index, out_operand_index);
  }

  auto &operand_li_map = _lowered_graph.lower_info().operand;

  // Find Permute information
  auto input_factor = operand_li_map.getRawPtr(operand_index)->def_factors().getOnlyElement();
  auto input_backend = input_factor.backend();
  auto output_backend = factor.backend();
  // NOTE Permute may not have specific layout because the layout of input and output may be
  // different.
  const auto permute_node_layout = ir::Layout::UNKNOWN;
  // NOTE If one backend supports several layout, the backend must support Permute operation
  const backend::Backend *permute_node_backend = compiler::BackendManager::get().getBuiltin();
  assert(permute_node_backend->config()->id() == onert::backend::builtin::Config::ID);

  if (input_backend == output_backend)
  {
    permute_node_backend = input_backend;
  }
  const PermuteFactor permute_node_factor{permute_node_backend, permute_node_layout};

  // Update LowerInfo of input operand
  auto operand_lower_info = operand_li_map.getRawPtr(operand_index);
  operand_lower_info->removeUsePermuteFactor(factor);
  operand_lower_info->addUsePermuteFactor(permute_node_factor);

  // Update LowerInfo of output operand
  auto out_operand_li = std::make_unique<compiler::OperandLowerInfo>();

  // The input and output factors of all nodes will be the same except Permute. So Tensor's
  // allocators allocates memory using only the information of def permutation factor now.
  // TODO Change param to permute_node_factor
  out_operand_li->addDefPermuteFactor(factor);
  out_operand_li->addUsePermuteFactor(factor);
  operand_li_map.set(out_operand_index, std::move(out_operand_li));

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

  VERBOSE_F() << "Permute Op inserted, node index : " << node_index << std::endl;
  VERBOSE_F() << "  - Input (original) Operand : " << operand_index << "("
              << input_factor.backend()->config()->id() << ")" << std::endl;
  VERBOSE_F() << "  - Output(inserted) Operand : " << out_operand_index << "("
              << factor.backend()->config()->id() << ")" << std::endl;

  // Operation LowerInfo
  {
    auto &operation_li_map = _lowered_graph.lower_info().operation;
    operation_li_map.set(node_index, std::make_unique<compiler::OperationLowerInfo>(
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
