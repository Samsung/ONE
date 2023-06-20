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

#include "PermutationOperationPass.h"

#include "backend/Backend.h"
#include "backend/IConfig.h"
#include "ir/Graph.h"
#include "util/logging.h"

namespace onert
{
namespace compiler
{
namespace pass
{

using namespace ir;

void PermutationOperationPass::callback(const OperationIndex &, IOperation &node)
{
  node.accept(*this);
}

// TODO Remove this. Expanding ranks of Operand is dangerous
void PermutationOperationPass::applyExpandRanks(const Operation &node)
{
  const auto &output_ind = node.getOutputs().at(0);
  const auto &output = _graph.operands().at(output_ind);

  assert(output.getDef().valid());
  const auto node_index = output.getDef();
  const auto frontend_layout = _graph.layout();
  const auto backend_layout = _lowered_graph.lower_info().operation.getRawPtr(node_index)->layout();

  if (frontend_layout == backend_layout)
  {
    return;
  }

  int32_t expanded_rank = 0;
  for (const auto &index :
       (node.getInputs() + node.getOutputs()) | Remove::DUPLICATED | Remove::UNDEFINED)
  {
    expanded_rank = std::max(expanded_rank, _graph.operands().at(index).shape().rank());
  }
  if (expanded_rank < 4)
    return;

  for (const auto &index :
       (node.getInputs() + node.getOutputs()) | Remove::DUPLICATED | Remove::UNDEFINED)
  {
    const auto &operand = _graph.operands().at(index);
    if (operand.shape().rank() < expanded_rank)
    {
      if (operand.getUses().size() > 1)
        throw std::runtime_error("PermutationOperationPass: not supported expanding rank of "
                                 "operand used in more than one node");
      // TODO remove const_cast later. For example, _ctx may need to be a non const variable or
      //      a node to extend shape may be inserted in front of this operation
      const_cast<Shape &>(operand.shape()).extendRank(expanded_rank);
    }
  }
}

void PermutationOperationPass::changeToKeepLayout(const Operation &node)
{
  const auto &output_ind = node.getOutputs().at(0);
  const auto &output_obj = _graph.operands().at(output_ind);

  assert(output_obj.getDef().valid());
  const auto node_index = output_obj.getDef();

  auto &operation_li_map = _lowered_graph.lower_info().operation;
  auto &operand_li_map = _lowered_graph.lower_info().operand;
  const auto frontend_layout = _graph.layout();
  const auto backend_layout = operation_li_map.getRawPtr(node_index)->layout();

  if (frontend_layout == backend_layout)
  {
    return;
  }

  // Permutation changing layout beyond 4-D is not supported yet
  assert(output_obj.shape().rank() <= 4);

  // Change PermuteFactors of operands and the operation of target node
  {
    const auto op_li = operation_li_map.getRawPtr(node_index);
    const auto backend = op_li->backend();

    operation_li_map.set(node_index,
                         std::make_unique<compiler::OperationLowerInfo>(backend, frontend_layout));

    const PermuteFactor removed_factor{backend, backend_layout};
    const PermuteFactor new_factor{backend, frontend_layout};
    for (const auto &input : node.getInputs() | Remove::DUPLICATED | Remove::UNDEFINED)
    {
      // Check if it can be removed by checking if the operand is used by another operation and
      // it uses the same backend and layout
      bool canRemove = true;
      for (const auto &use : _graph.operands().at(input).getUses())
      {
        if (use != node_index)
        {
          auto use_op_li = operation_li_map.getRawPtr(use);
          if (use_op_li->backend() == backend && use_op_li->layout() == backend_layout)
          {
            canRemove = false;
            break;
          }
        }
      }

      auto input_li = operand_li_map.getRawPtr(input);
      if (canRemove)
      {
        input_li->removeUsePermuteFactor(removed_factor);
      }
      input_li->addUsePermuteFactor(new_factor);

      // Whether if node's input is an input of model or a constant
      if (!_graph.operands().at(input).getDef().valid() &&
          (input_li->def_factors().size() == 1 &&
           input_li->def_factors().getOnlyElement() == removed_factor))
      {
        assert(_graph.getInputs().contains(input) || _graph.operands().at(input).isConstant());
        input_li->removeDefPermuteFactor(removed_factor);
        input_li->addDefPermuteFactor(new_factor);
      }
    }

    for (const auto &output : node.getOutputs() | Remove::DUPLICATED | Remove::UNDEFINED)
    {
      auto lower_info = operand_li_map.getRawPtr(output);
      lower_info->removeDefPermuteFactor(removed_factor);
      lower_info->addDefPermuteFactor(new_factor);

      // Whether if node's output is an output of model
      if (_graph.operands().at(output).getUses().size() == 0)
      {
        assert(_graph.getOutputs().contains(output));
        lower_info->removeUsePermuteFactor(removed_factor);
        lower_info->addUsePermuteFactor(new_factor);
      }
    }
  }
}

void PermutationOperationPass::visit(const ir::operation::BinaryArithmetic &node)
{
  applyExpandRanks(node);
}

void PermutationOperationPass::visit(const ir::operation::Concat &node) { applyExpandRanks(node); }

void PermutationOperationPass::visit(const ir::operation::Comparison &node)
{
  applyExpandRanks(node);
}

void PermutationOperationPass::visit(const ir::operation::ElementwiseBinary &node)
{
  applyExpandRanks(node);
}

void PermutationOperationPass::visit(const ir::operation::ElementwiseUnary &node)
{
  applyExpandRanks(node);
}

void PermutationOperationPass::visit(const ir::operation::FullyConnected &node)
{
  const auto &input_ind = node.getInputs().at(ir::operation::FullyConnected::Input::INPUT);
  const auto &input_obj = _graph.operands().at(input_ind);
  const auto &input_shape = input_obj.shape();

  if (input_shape.rank() >= 4)
  {
    changeToKeepLayout(node);
  }
}

void PermutationOperationPass::visit(const ir::operation::Gather &node)
{
  const auto &input_ind = node.getInputs().at(ir::operation::Gather::Input::INPUT);
  const auto &input_obj = _graph.operands().at(input_ind);
  const auto &input_shape = input_obj.shape();

  const auto &output_ind = node.getOutputs().at(0);
  const auto &output_obj = _graph.operands().at(output_ind);
  const auto &output_shape = output_obj.shape();

  if (input_shape.rank() >= 4 || output_shape.rank() >= 4)
  {
    changeToKeepLayout(node);
  }
}

void PermutationOperationPass::visit(const ir::operation::OneHot &node)
{
  const auto &output_ind = node.getOutputs().at(0);
  const auto &output_obj = _graph.operands().at(output_ind);
  const auto &output_shape = output_obj.shape();

  if (output_shape.rank() >= 4)
  {
    changeToKeepLayout(node);
  }
}

void PermutationOperationPass::visit(const ir::operation::Pack &node)
{
  const auto &input_ind = node.getInputs().at(ir::operation::Reshape::Input::INPUT);
  const auto &input_obj = _graph.operands().at(input_ind);
  const auto &input_shape = input_obj.shape();

  const auto &output_ind = node.getOutputs().at(0);
  const auto &output_obj = _graph.operands().at(output_ind);
  const auto &output_shape = output_obj.shape();

  if (input_shape.rank() < 4 || output_shape.rank() >= 4)
  {
    changeToKeepLayout(node);
  }
}

void PermutationOperationPass::visit(const ir::operation::PReLU &node) { applyExpandRanks(node); }

void PermutationOperationPass::visit(const ir::operation::Reshape &node)
{
  const auto &input_ind = node.getInputs().at(ir::operation::Reshape::Input::INPUT);
  const auto &input_obj = _graph.operands().at(input_ind);
  const auto &input_shape = input_obj.shape();

  const auto &output_ind = node.getOutputs().at(0);
  const auto &output_obj = _graph.operands().at(output_ind);
  const auto &output_shape = output_obj.shape();

  if (input_shape.rank() >= 4 || output_shape.rank() >= 4)
  {
    changeToKeepLayout(node);
  }
}

void PermutationOperationPass::visit(const ir::operation::SquaredDifference &node)
{
  applyExpandRanks(node);
}

void PermutationOperationPass::visit(const ir::operation::Unpack &node)
{
  const auto &input_ind = node.getInputs().at(ir::operation::Reshape::Input::INPUT);
  const auto &input_obj = _graph.operands().at(input_ind);
  const auto &input_shape = input_obj.shape();

  const auto &output_ind = node.getOutputs().at(0);
  const auto &output_obj = _graph.operands().at(output_ind);
  const auto &output_shape = output_obj.shape();

  if (input_shape.rank() < 4 || output_shape.rank() >= 4)
  {
    changeToKeepLayout(node);
  }
}

} // namespace pass
} // namespace compiler
} // namespace onert
