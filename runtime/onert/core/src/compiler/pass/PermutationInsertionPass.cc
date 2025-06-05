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

#include "ir/operation/Permute.h"
#include "util/logging.h"

#include <cassert>
#include <memory>
#include <unordered_map>
#include <utility>

namespace onert::compiler::pass
{

void PermutationInsertionPass::callback(const ir::OperandIndex &index, ir::Operand &object)
{
  auto &operand_li_map = _lowered_graph.lower_info().operand;
  auto &&operand_li = operand_li_map.getRawPtr(index);
  assert(operand_li);

  // NOTE Later, constants also will have Def
  // Ignore constants
  if (operand_li->def_backends().size() == 0)
  {
    return;
  }

  std::list<ir::OperationIndex> permute_indexes;

  // Build a map for all necessary type of operands
  std::unordered_map<const backend::Backend *, ir::OperandIndex> backend_to_index;
  {
    assert(operand_li->def_backends().size() == 1);
    for (auto &&backend : operand_li->def_backends())
    {
      backend_to_index.emplace(backend, index);
    }

    auto insert_set = operand_li->use_backends() - operand_li->def_backends();
    for (auto &&backend : insert_set)
    {
      const auto permute_operation_index = insertPermute(index, backend);
      permute_indexes.push_back(permute_operation_index);
      const auto &permute_operation = _graph.operations().at(permute_operation_index);
      const auto permuted_operand_index = permute_operation.getOutputs().at(0);
      backend_to_index.emplace(backend, permuted_operand_index);
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
      const auto backend = _lowered_graph.lower_info().operation.at(use);
      assert(backend);
      assert(operation.getInputs().contains(index));

      auto new_index = backend_to_index.at(backend);
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
                                                           const backend::Backend *backend)
{
  auto &operand = _graph.operands().at(operand_index);

  // Generate output operand and permute operation
  auto out_operand_index = _graph.addOperand(operand.shape(), operand.typeInfo());
  // change model output if operand_index is model output index and the out operand is builtin
  // backend
  auto &model_outputs = _graph.getOutputs();
  const backend::Backend *builtin_backend = compiler::BackendManager::get().getBuiltin();
  assert(builtin_backend->config()->id() == onert::backend::builtin::Config::ID);

  if (model_outputs.contains(operand_index) && backend == builtin_backend)
  {
    model_outputs.replace(operand_index, out_operand_index);
  }

  auto &operand_li_map = _lowered_graph.lower_info().operand;

  // Find Permute information
  auto input_backend = operand_li_map.getRawPtr(operand_index)->def_backends().getOnlyElement();
  const backend::Backend *permute_node_backend = compiler::BackendManager::get().getBuiltin();
  assert(permute_node_backend->config()->id() == onert::backend::builtin::Config::ID);
  assert(input_backend != backend);

  // Update LowerInfo of input operand
  auto operand_lower_info = operand_li_map.getRawPtr(operand_index);
  operand_lower_info->removeUseBackend(backend);
  operand_lower_info->addUseBackend(permute_node_backend);

  // Update LowerInfo of output operand
  auto out_operand_li = std::make_unique<compiler::OperandLowerInfo>();

  // The input and output backends of all nodes will be the same except Permute. So Tensor's
  // allocators allocates memory using only the information of def permutation backend now.
  // TODO Change param to permute_node_factor
  out_operand_li->addDefBackend(backend);
  out_operand_li->addUseBackend(backend);
  operand_li_map.set(out_operand_index, std::move(out_operand_li));

  // Insert permute operation to the graph
  using Permute = ir::operation::Permute;
  auto insert_node =
    std::make_unique<Permute>(operand_index, out_operand_index, ir::PermuteType::COPY);

  auto node_index = _graph.operations().push(std::move(insert_node));

  VERBOSE_F() << "Permute Op inserted, node index : " << node_index << std::endl;
  VERBOSE_F() << "  - Input (original) Operand : " << operand_index << "("
              << input_backend->config()->id() << ")" << std::endl;
  VERBOSE_F() << "  - Output(inserted) Operand : " << out_operand_index << "("
              << backend->config()->id() << ")" << std::endl;

  // Operation LowerInfo
  {
    auto &operation_li_map = _lowered_graph.lower_info().operation;
    operation_li_map.emplace(node_index, permute_node_backend);
  }

  // Update Use/Def info
  {
    _graph.operands().at(operand_index).insertUse(node_index);
    _graph.operands().at(out_operand_index).setDef(node_index);
  }
  return node_index;
}
} // namespace onert::compiler::pass
