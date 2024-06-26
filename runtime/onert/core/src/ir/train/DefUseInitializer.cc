/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "DefUseInitializer.h"

#include "ir/train/Index.h"
#include "../verifier/Verifier.h"

#include <cassert>
#include <memory>

// TODO Reduce duplicate code

namespace onert
{
namespace ir
{
namespace train
{

DefUseInitializer::DefUseInitializer(TrainableGraph &tgraph)
  : _tgraph{tgraph}, _node_to_idx{}, _training_defuses{}
{
  for (const auto &index : _tgraph.topolSortOperations())
  {
    const auto &node = _tgraph.operation(index);
    assert(_node_to_idx.find(&node) == _node_to_idx.end());
    _node_to_idx[&node] = index;
  }
}

void DefUseInitializer::operator()()
{
  const auto &graph = _tgraph.graph();
  assert(ir::verifier::EdgeChecker().verify(graph));

  _training_defuses.clear();
  graph.operands().iterate([&](const ir::OperandIndex &idx, const ir::Operand &operand) {
    // Initialize as emtpy DefUseChain
    const auto empty_defuse_chain = DefUseChain{operand};
    _training_defuses.emplace(TrainingOperandIndex{idx, true}, empty_defuse_chain);
    _training_defuses.emplace(TrainingOperandIndex{idx, false}, empty_defuse_chain);
  });

  // Initialize training def-uses of forwarding operands for only forwarding nodes
  // (i.e. forwarding nodes that do not have any backwarding node)
  graph.operands().iterate([&](const ir::OperandIndex &idx, const ir::Operand &operand) {
    // Append forwarding def-uses as it is
    const bool is_forward = true;
    const auto forwarding_operand_index = TrainingOperandIndex{idx, is_forward};

    const auto def = operand.getDef();
    auto &defuse_chain = _training_defuses.at(forwarding_operand_index);
    defuse_chain.setTrainingDef(TrainingOperationIndex{def, is_forward});

    assert(_training_defuses.at(forwarding_operand_index).getTrainingUses().size() == 0);
    const auto uses = operand.getUses();
    for (const auto &use : uses)
      insertUse(forwarding_operand_index, TrainingOperationIndex{use, is_forward});
  });

  // Initialize training uses of forwarding operands and def-uses of backwarding operands for
  // backwarding nodes (i.e. backwarding nodes that do not have any forwarding node)
  auto backward_order = _tgraph.btopolSortOperations();
  // get rid of all nodes not reachable from a node with trainable parameters
  // backward_order = _tgraph.truncateBackwardOrder(backward_order);
  for (const auto &op_index : backward_order)
  {
    const auto &node = _tgraph.operation(op_index);

    // Insert use of backwarding operands(only output)
    {
      if (node.getOutputs().size() > 1)
        throw std::runtime_error(
          "DefUseInitializer does not support multiple outputs of training operation");

      const auto &output = node.getOutputs().at(0);
      const auto backwarding_op_index = TrainingOperationIndex{op_index, false};
      const auto backwarding_operand_index = TrainingOperandIndex{output, false};
      insertUse(backwarding_operand_index, backwarding_op_index);
    }

    // TODO Apply `node.isRequiredForBackward()`
    // Insert uses of forwarding operands and set def of backwarding operands
    node.accept(*this);
  }

  _tgraph.setTrainingDefUses(_training_defuses);
}

void DefUseInitializer::visit(const train::operation::BinaryArithmetic &node)
{
  assert(_node_to_idx.find(&node) != _node_to_idx.end());
  const auto &op_index = _node_to_idx.at(&node);
  const auto backwarding_op_index = TrainingOperationIndex{op_index, false};

  // Insert uses of forwarding output
  if (node.param().activation != ir::Activation::NONE)
  {
    const auto &out_index = node.getOutputs().at(0);
    const auto out_forwarding_index = TrainingOperandIndex{out_index, true};
    insertUse(out_forwarding_index, backwarding_op_index);
  }

  for (const auto &in_index : node.getInputs() | ir::Remove::UNDEFINED | ir::Remove::DUPLICATED)
  {
    // Insert use of forwarding inputs
    const auto in_forwarding_index = TrainingOperandIndex{in_index, true};
    insertUse(in_forwarding_index, backwarding_op_index);

    // Set def of backwarding(backprop) inputs
    const auto in_back_prop_index = TrainingOperandIndex{in_index, false};
    setBackPropDef(in_back_prop_index, backwarding_op_index);
  }
}

void DefUseInitializer::visit(const train::operation::Conv2D &node)
{
  assert(_node_to_idx.find(&node) != _node_to_idx.end());
  const auto &op_index = _node_to_idx.at(&node);
  const auto backwarding_op_index = TrainingOperationIndex{op_index, false};

  // Insert use of forwarding inputs
  const auto &in_index = node.getInputs().at(train::operation::Conv2D::Input::INPUT);
  const auto in_forwarding_index = TrainingOperandIndex{in_index, true};
  insertUse(in_forwarding_index, backwarding_op_index);

  const auto &weights_index = node.getInputs().at(train::operation::Conv2D::Input::KERNEL);
  const auto weights_forwarding_index = TrainingOperandIndex{weights_index, true};
  insertUse(weights_forwarding_index, backwarding_op_index);
  // Bias is not used in backwarding op

  // Insert use of forwarding output
  if (node.param().activation != ir::Activation::NONE)
  {
    const auto &out_index = node.getOutputs().at(0);
    const auto out_forwarding_index = TrainingOperandIndex{out_index, true};
    insertUse(out_forwarding_index, backwarding_op_index);
  }

  // Set def of backwarding inputs
  const auto in_backwarding_index = TrainingOperandIndex{in_index, false};
  setBackPropDef(in_backwarding_index, backwarding_op_index);

  const auto weights_backwarding_index = TrainingOperandIndex{weights_index, false};
  setDef(weights_backwarding_index, backwarding_op_index);

  const auto &bias_index = node.getInputs().at(ir::operation::Conv2D::Input::BIAS);
  if (bias_index.valid())
  {
    const auto bias_backwarding_index = TrainingOperandIndex{bias_index, false};
    setDef(bias_backwarding_index, backwarding_op_index);
  }
}

void DefUseInitializer::visit(const train::operation::DepthwiseConv2D &node)
{
  assert(_node_to_idx.find(&node) != _node_to_idx.end());
  const auto &op_index = _node_to_idx.at(&node);
  const auto backwarding_op_index = TrainingOperationIndex{op_index, false};

  // Insert use of forwarding inputs
  const auto &in_index = node.getInputs().at(train::operation::DepthwiseConv2D::Input::INPUT);
  const auto in_forwarding_index = TrainingOperandIndex{in_index, true};
  insertUse(in_forwarding_index, backwarding_op_index);

  const auto &weights_index = node.getInputs().at(train::operation::DepthwiseConv2D::Input::KERNEL);
  const auto weights_forwarding_index = TrainingOperandIndex{weights_index, true};
  insertUse(weights_forwarding_index, backwarding_op_index);
  // Bias is not used in backwarding op

  // Insert uses of forwarding output
  if (node.param().activation != ir::Activation::NONE)
  {
    const auto &out_index = node.getOutputs().at(0);
    const auto out_forwarding_index = TrainingOperandIndex{out_index, true};
    insertUse(out_forwarding_index, backwarding_op_index);
  }

  // Set def of backwarding inputs
  const auto in_backwarding_index = TrainingOperandIndex{in_index, false};
  setBackPropDef(in_backwarding_index, backwarding_op_index);

  const auto weights_backwarding_index = TrainingOperandIndex{weights_index, false};
  setDef(weights_backwarding_index, backwarding_op_index);

  const auto &bias_index = node.getInputs().at(ir::operation::Conv2D::Input::BIAS);
  if (bias_index.valid())
  {
    const auto bias_backwarding_index = TrainingOperandIndex{bias_index, false};
    setDef(bias_backwarding_index, backwarding_op_index);
  }
}

void DefUseInitializer::visit(const train::operation::ElementwiseActivation &node)
{
  if (node.param().op_type != operation::ElementwiseActivation::Type::RELU)
  {
    throw std::runtime_error{"DefUseInitializer: Not yet supported activation type"};
  }
  assert(_node_to_idx.find(&node) != _node_to_idx.end());
  const auto &op_index = _node_to_idx.at(&node);
  const auto backwarding_op_index = TrainingOperationIndex{op_index, false};

  // Insert use of forwarding output
  const auto &out_index = node.getOutputs().at(0);
  const auto out_forwarding_index = TrainingOperandIndex{out_index, true};
  insertUse(out_forwarding_index, backwarding_op_index);

  // Set def of backwarding(backprop) inputs
  for (const auto &in_index : node.getInputs() | ir::Remove::UNDEFINED | ir::Remove::DUPLICATED)
  {
    const auto in_back_prop_index = TrainingOperandIndex{in_index, false};
    setBackPropDef(in_back_prop_index, backwarding_op_index);
  }
}

void DefUseInitializer::visit(const train::operation::FullyConnected &node)
{
  assert(_node_to_idx.find(&node) != _node_to_idx.end());
  const auto &op_index = _node_to_idx.at(&node);
  const auto backwarding_op_index = TrainingOperationIndex{op_index, false};

  // Insert use of forwarding inputs
  const auto &in_index = node.getInputs().at(train::operation::FullyConnected::Input::INPUT);
  const auto in_forwarding_index = TrainingOperandIndex{in_index, true};
  insertUse(in_forwarding_index, backwarding_op_index);

  const auto &weights_index = node.getInputs().at(train::operation::FullyConnected::Input::WEIGHT);
  const auto weights_forwarding_index = TrainingOperandIndex{weights_index, true};
  insertUse(weights_forwarding_index, backwarding_op_index);
  // Bias is not used in backwarding op

  // Insert uses of forwarding output
  if (node.param().activation != ir::Activation::NONE)
  {
    const auto &out_index = node.getOutputs().at(0);
    const auto out_forwarding_index = TrainingOperandIndex{out_index, true};
    insertUse(out_forwarding_index, backwarding_op_index);
  }

  // Set def of backwarding inputs
  const auto in_backwarding_index = TrainingOperandIndex{in_index, false};
  setBackPropDef(in_backwarding_index, backwarding_op_index);

  const auto weights_backwarding_index = TrainingOperandIndex{weights_index, false};
  setDef(weights_backwarding_index, backwarding_op_index);

  const auto &bias_index = node.getInputs().at(ir::operation::Conv2D::Input::BIAS);
  if (bias_index.valid())
  {
    const auto bias_backwarding_index = TrainingOperandIndex{bias_index, false};
    setDef(bias_backwarding_index, backwarding_op_index);
  }
}

void DefUseInitializer::visit(const train::operation::Loss &node)
{
  assert(_node_to_idx.find(&node) != _node_to_idx.end());
  const auto &op_index = _node_to_idx.at(&node);
  const auto backwarding_op_index = TrainingOperationIndex{op_index, false};

  for (const auto &in_index : node.getInputs() | ir::Remove::UNDEFINED | ir::Remove::DUPLICATED)
  {
    // Insert use of forwarding inputs
    const auto in_forwarding_index = TrainingOperandIndex{in_index, true};
    insertUse(in_forwarding_index, backwarding_op_index);
  }

  // Set def of backwarding(backprop) y_pred
  const auto &y_pred_index = node.getInputs().at(train::operation::Loss::Input::Y_PRED);
  assert(!_tgraph.operands().at(y_pred_index).isConstant());
  const auto y_pred_back_prop_index = TrainingOperandIndex{y_pred_index, false};
  setBackPropDef(y_pred_back_prop_index, backwarding_op_index);

  // Set def of backwarding(backprop) y_true
  const auto &y_true_index = node.getInputs().at(train::operation::Loss::Input::Y_TRUE);
  assert(!_tgraph.operands().at(y_true_index).isConstant());
  const auto y_true_back_prop_index = TrainingOperandIndex{y_true_index, false};
  setBackPropDef(y_true_back_prop_index, backwarding_op_index);

  // Remove use of backwarding output
  const auto &out_index = node.getOutputs().at(0);
  const auto out_backwarding_index = TrainingOperandIndex{out_index, false};
  auto &defuse_chain = _training_defuses.at(out_backwarding_index);
  defuse_chain.removeTrainingUse(backwarding_op_index);
}

void DefUseInitializer::visit(const train::operation::Pad &node)
{
  assert(_node_to_idx.find(&node) != _node_to_idx.end());
  const auto &op_index = _node_to_idx.at(&node);
  const auto backwarding_op_index = TrainingOperationIndex{op_index, false};

  // Insert use of forwarding pad
  const auto &pad_index = node.getInputs().at(train::operation::Pad::Input::PAD);
  const auto pad_forwarding_index = TrainingOperandIndex{pad_index, true};
  insertUse(pad_forwarding_index, backwarding_op_index);

  // Insert use of backwarding(backprop) output
  const auto &out_index = node.getOutputs().at(0);
  const auto out_backwarding_index = TrainingOperandIndex{out_index, false};
  insertUse(out_backwarding_index, backwarding_op_index);

  // Set def of backwarding(backprop) input
  const auto &in_index = node.getInputs().at(train::operation::Pad::Input::INPUT);
  const auto in_back_prop_index = TrainingOperandIndex{in_index, false};
  setBackPropDef(in_back_prop_index, backwarding_op_index);
}

void DefUseInitializer::visit(const train::operation::Pool2D &node)
{
  if (node.param().op_type != ir::operation::Pool2D::PoolType::MAX)
  {
    throw std::runtime_error{"DefUseInitializer: Not yet supported pool type"};
  }

  assert(_node_to_idx.find(&node) != _node_to_idx.end());
  const auto &op_index = _node_to_idx.at(&node);
  const auto backwarding_op_index = TrainingOperationIndex{op_index, false};

  // Insert uses of forwarding output
  if (node.param().activation != ir::Activation::NONE)
  {
    const auto &out_index = node.getOutputs().at(0);
    const auto out_forwarding_index = TrainingOperandIndex{out_index, true};
    insertUse(out_forwarding_index, backwarding_op_index);
  }

  // Insert use of backwarding(backprop) output
  const auto &out_index = node.getOutputs().at(0);
  const auto out_backwarding_index = TrainingOperandIndex{out_index, false};
  insertUse(out_backwarding_index, backwarding_op_index);

  // Set def of backwarding(backprop) input
  const auto &in_index = node.getInputs().at(train::operation::Pool2D::Input::INPUT);
  const auto in_back_prop_index = TrainingOperandIndex{in_index, false};
  setBackPropDef(in_back_prop_index, backwarding_op_index);
}

void DefUseInitializer::visit(const train::operation::Reduce &node)
{
  if (node.param().reduce_type != ir::operation::Reduce::ReduceType::MEAN)
  {
    throw std::runtime_error{"DefUseInitializer: Not yet supported reduce type"};
  }

  assert(_node_to_idx.find(&node) != _node_to_idx.end());
  const auto &op_index = _node_to_idx.at(&node);
  const auto backwarding_op_index = TrainingOperationIndex{op_index, false};

  // Insert use of backwarding(backprop) output
  const auto &out_index = node.getOutputs().at(0);
  const auto out_backwarding_index = TrainingOperandIndex{out_index, false};
  insertUse(out_backwarding_index, backwarding_op_index);

  // Set def of backwarding(backprop) input
  const auto &in_index = node.getInputs().at(train::operation::Reduce::Input::INPUT);
  const auto in_back_prop_index = TrainingOperandIndex{in_index, false};
  setBackPropDef(in_back_prop_index, backwarding_op_index);
}

void DefUseInitializer::visit(const train::operation::Reshape &node)
{
  assert(_node_to_idx.find(&node) != _node_to_idx.end());
  const auto &op_index = _node_to_idx.at(&node);
  const auto backwarding_op_index = TrainingOperationIndex{op_index, false};

  // Insert uses of forwarding output
  const auto &out_index = node.getOutputs().at(0);
  const auto out_forwarding_index = TrainingOperandIndex{out_index, true};
  insertUse(out_forwarding_index, backwarding_op_index);

  // Insert use of backwarding(backprop) output
  const auto out_backwarding_index = TrainingOperandIndex{out_index, false};
  insertUse(out_backwarding_index, backwarding_op_index);

  // Set def of backwarding(backprop) input
  const auto &in_index = node.getInputs().at(train::operation::Reduce::Input::INPUT);
  const auto in_back_prop_index = TrainingOperandIndex{in_index, false};
  setBackPropDef(in_back_prop_index, backwarding_op_index);
}

void DefUseInitializer::visit(const train::operation::Softmax &node)
{
  assert(_node_to_idx.find(&node) != _node_to_idx.end());
  const auto &op_index = _node_to_idx.at(&node);
  const auto backwarding_op_index = TrainingOperationIndex{op_index, false};

  // Insert uses of forwarding output
  const auto &out_index = node.getOutputs().at(0);
  const auto out_forwarding_index = TrainingOperandIndex{out_index, true};
  insertUse(out_forwarding_index, backwarding_op_index);

  // Insert use of backwarding(backprop) output
  const auto out_backwarding_index = TrainingOperandIndex{out_index, false};
  insertUse(out_backwarding_index, backwarding_op_index);

  // Set def of backwarding(backprop) input
  const auto &in_index = node.getInputs().at(train::operation::Reduce::Input::INPUT);
  const auto in_back_prop_index = TrainingOperandIndex{in_index, false};
  setBackPropDef(in_back_prop_index, backwarding_op_index);
}

void DefUseInitializer::insertUse(const TrainingOperandIndex &operand_index,
                                  const TrainingOperationIndex &op_index)
{
  assert(_training_defuses.find(operand_index) != _training_defuses.end());
  auto &defuse_chain = _training_defuses.at(operand_index);
  defuse_chain.insertTrainingUse(op_index);
}

void DefUseInitializer::setDef(const TrainingOperandIndex &operand_index,
                               const TrainingOperationIndex &op_index)
{
  assert(operand_index.valid());

  auto &defuse_chain = _training_defuses.at(operand_index);
  assert(!defuse_chain.getTrainingDef().valid());
  defuse_chain.setTrainingDef(op_index);
}

void DefUseInitializer::setBackPropDef(const TrainingOperandIndex &operand_index,
                                       const TrainingOperationIndex &op_index)
{
  // NOTE There is no need to set def of constant backwarding(backprop) inputs
  //      because it won't be back-propagated.
  if (!_tgraph.operands().at(operand_index.index()).isConstant())
  {
    setDef(operand_index, op_index);
  }
}

} // namespace train
} // namespace ir
} // namespace onert
