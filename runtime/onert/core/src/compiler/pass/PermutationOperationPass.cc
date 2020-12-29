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

void PermutationOperationPass::callback(const OperationIndex &, Operation &node)
{
  node.accept(*this);
};

// TODO Remove this. Expanding ranks of Operand is dangerous
void PermutationOperationPass::applyExpandRanks(const Operation &node)
{
  const auto &output_ind = node.getOutputs().at(0);
  const auto &output = _graph.operands().at(output_ind);

  assert(output.getDef().valid());
  const auto node_index = output.getDef();
  const auto &op_seq_index = _lowered_graph.op_seqs().getOperation(node_index);
  const auto frontend_layout = _lowered_graph.op_seqs().at(op_seq_index).getLayout();
  const auto backend_layout = _lowered_graph.getLowerInfo(op_seq_index)->layout();

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
  const auto &op_seq_index = _lowered_graph.op_seqs().getOperation(node_index);

  const auto frontend_layout = _lowered_graph.op_seqs().at(op_seq_index).getLayout();
  const auto backend_layout = _lowered_graph.getLowerInfo(op_seq_index)->layout();

  if (frontend_layout == backend_layout)
  {
    return;
  }

  // Permutation changing layout beyond 4-D is not supported yet
  assert(output_obj.shape().rank() <= 4);

  // Divide op_seq based on target operation
  {
    auto &prev_op_seq = _lowered_graph.op_seqs().at(op_seq_index);
    auto &operations = _lowered_graph.graph().operations();

    // Create new op_seq and move information from existing op_seq to new op_seq if target
    // node is the end of op_seq
    auto it = prev_op_seq.begin();
    // Find iterator of target node in op_seq
    while (*(it++) != node_index)
      ;
    if (it != prev_op_seq.end())
    {
      const auto &target_op_idx = *it;
      const auto &target_node = operations.at(target_op_idx);
      const auto &next_op_seq_index =
        _lowered_graph.op_seqs().emplace(target_op_idx, prev_op_seq.getLayout());
      auto &next_op_seq = _lowered_graph.op_seqs().at(next_op_seq_index);
      next_op_seq.setInputs(target_node.getInputs());
      next_op_seq.setOutputs(target_node.getOutputs());

      std::vector<OperationIndex> remove_list;
      remove_list.emplace_back(target_op_idx);
      while (++it != prev_op_seq.end())
      {
        next_op_seq.appendOperation(target_op_idx);
        next_op_seq.setOutputs(target_node.getOutputs());
        remove_list.emplace_back(target_op_idx);
      }

      prev_op_seq.setOutputs(node.getOutputs());
      for (const auto &index : remove_list)
      {
        prev_op_seq.remove(index);
      }

      const auto op_seq_li = _lowered_graph.getLowerInfo(op_seq_index);
      _lowered_graph.setLowerInfo(
        next_op_seq_index,
        std::make_unique<compiler::OpSequenceLowerInfo>(op_seq_li->backend(), op_seq_li->layout()));
    }
  }

  // Remove target operation from op_seq and insert the target operation to new op_seq
  {
    const auto backend = _lowered_graph.getLowerInfo(op_seq_index)->backend();

    // Remove target operation from op_sequence
    _lowered_graph.op_seqs().removeFromOpSequence(node_index);

    if (!_lowered_graph.op_seqs().exist(op_seq_index))
    {
      // Remove lowerinfo for op_seq of target operation if the op_seq does not exist
      _lowered_graph.removeLowerInfo(op_seq_index);
    }
    else
    {
      // Update op_seq of target operation if the op_seq exists
      auto &prev_op_seq = _lowered_graph.op_seqs().at(op_seq_index);
      const auto &last_node_idx = *(--prev_op_seq.end());
      const auto &last_node = _lowered_graph.graph().operations().at(last_node_idx);
      prev_op_seq.setOutputs(last_node.getOutputs());
    }

    // Create new op_seq and set information to the op_seq
    auto new_op_seq_index = _lowered_graph.op_seqs().emplace(node_index, frontend_layout);
    auto &new_op_seq = _lowered_graph.op_seqs().at(new_op_seq_index);
    new_op_seq.setInputs(node.getInputs());
    new_op_seq.setOutputs(node.getOutputs());
    _lowered_graph.setLowerInfo(
      new_op_seq_index, std::make_unique<compiler::OpSequenceLowerInfo>(backend, frontend_layout));
  }

  // Change PermuteFactors of operands of target node
  {
    const auto &op_seq_index = _lowered_graph.op_seqs().getOperation(node_index);
    const auto op_seq_li = _lowered_graph.getLowerInfo(op_seq_index);
    const auto backend = op_seq_li->backend();
    const PermuteFactor removed_factor{backend, backend_layout};
    const PermuteFactor new_factor{backend, frontend_layout};
    for (const auto &input : node.getInputs() | Remove::DUPLICATED | Remove::UNDEFINED)
    {
      bool canRemove = true;
      for (const auto &use : _graph.operands().at(input).getUses())
      {
        if (use != node_index)
        {
          const auto &use_op_seq_index = _lowered_graph.op_seqs().getOperation(use);
          auto use_op_seq_li = _lowered_graph.getLowerInfo(use_op_seq_index);
          if (use_op_seq_li->backend() == backend && use_op_seq_li->layout() == backend_layout)
          {
            canRemove = false;
            break;
          }
        }
      }

      auto lower_info = _lowered_graph.getLowerInfo(input);
      if (canRemove)
      {
        lower_info->removeUsePermuteFactor(removed_factor);
      }
      lower_info->addUsePermuteFactor(new_factor);

      // Whether if node's input is an input of model or a constant
      if (!_graph.operands().at(input).getDef().valid() &&
          (lower_info->def_factors().size() == 1 &&
           lower_info->def_factors().getOnlyElement() == removed_factor))
      {
        assert(_graph.getInputs().contains(input) || _graph.operands().at(input).isConstant());
        lower_info->removeDefPermuteFactor(removed_factor);
        lower_info->addDefPermuteFactor(new_factor);
      }
    }

    for (const auto &output : node.getOutputs() | Remove::DUPLICATED | Remove::UNDEFINED)
    {
      auto lower_info = _lowered_graph.getLowerInfo(output);
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
