/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ModelEdgeInsertionPass.h"

#include "../../backend/builtin/Config.h"

#include "compiler/OperationLowerInfo.h"
#include "ir/operation/Permute.h"
#include "util/logging.h"

#include <cassert>
#include <memory>
#include <unordered_map>
#include <utility>

namespace
{

onert::ir::ModelEdge get_edge(const onert::ir::IOIndex &io_index,
                              const onert::ir::ModelEdgeSet &edge_set)
{
  onert::ir::ModelEdge ret;
  assert(std::get<2>(ret.from).undefined());

  for (const auto &edge : edge_set)
  {
    if (io_index == std::get<2>(edge.from))
    {
      ret = edge;
      break;
    }
  }

  return ret;
}

} // namespace

namespace onert
{
namespace compiler
{
namespace pass
{

ModelEdgeInsertionPass::ModelEdgeInsertionPass(compiler::LoweredGraph &lowered_graph,
                                               const ir::ModelEdgeSet &edge_set)
  : LoweredOperandPass{lowered_graph}, _edge_set{edge_set}
{
  // Validate edge_set
  if (edge_set.size() > 1)
  {
    const auto &first_from = edge_set.begin()->from;
    const auto &model_index = std::get<0>(first_from);
    const auto &subg_index = std::get<1>(first_from);

    for (const auto &edge : edge_set)
    {
      // Throw an error if edge_set has edges from different models
      if (model_index != std::get<0>(edge.from) || subg_index != std::get<1>(edge.from))
        throw std::runtime_error("ModelEdgeInsertionPass: Invalid edge");
    }
  }
}

void ModelEdgeInsertionPass::callback(const ir::OperandIndex &operand_index, ir::Operand &object)
{
  // Check if index is a output of subgraph
  const auto &outputs = _lowered_graph.graph().getOutputs();
  if (!_lowered_graph.graph().getOutputs().contains(operand_index))
    return;

  // Find IOIndex
  auto it = outputs.begin();
  while (it != outputs.end())
  {
    if (*it == operand_index)
      break;
    ++it;
  }
  ir::IOIndex io_index{it->value()};

  // Get model edge and check if edge set include the output
  auto edge = get_edge(io_index, _edge_set);
  if (std::get<2>(edge.from).undefined())
    return;

  // Insert ModelEdge operation
  ir::operation::ModelEdge::Param param;
  param.to_model_index = std::get<0>(edge.from);
  param.to_subg_index = std::get<1>(edge.from);
  param.to_input_index = std::get<2>(edge.from);

  auto new_node = std::make_unique<ir::operation::ModelEdge>(
    ir::OperandIndexSequence{operand_index}, ir::OperandIndexSequence{}, param);
  auto node_index = _graph.operations().push(std::move(new_node));

  // NOTE ModelEdge operation may not have specific layout because the layout of input and output
  // may be different.
  const auto node_layout = ir::Layout::UNKNOWN;
  const backend::Backend *node_backend = compiler::BackendManager::get().getBuiltin();

  auto &operand_li_map = _lowered_graph.lower_info().operand;
  auto input_factor = operand_li_map.getRawPtr(operand_index)->def_factors().getOnlyElement();

  VERBOSE_F() << "ModelEdge Op inserted, node index : " << node_index << std::endl;
  VERBOSE_F() << "  - Input (original) Operand : " << operand_index << "("
              << input_factor.backend()->config()->id() << ")" << std::endl;

  // Update LowerInfo of input operand
  const PermuteFactor node_factor{node_backend, node_layout};
  auto operand_lower_info = operand_li_map.getRawPtr(operand_index);
  operand_lower_info->addUsePermuteFactor(node_factor);

  // Operation LowerInfo
  {
    auto &operation_li_map = _lowered_graph.lower_info().operation;
    operation_li_map.set(node_index,
                         std::make_unique<compiler::OperationLowerInfo>(node_backend, node_layout));
  }

  // Update Use/Def info
  {
    object.insertUse(node_index);
  }
}

} // namespace pass
} // namespace compiler
} // namespace onert
