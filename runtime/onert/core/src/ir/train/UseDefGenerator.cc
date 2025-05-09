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

#include "UseDefGenerator.h"

#include "ir/train/TrainableGraph.h"
#include "ir/train/Index.h"
#include "../verifier/Verifier.h"

#include <cassert>
#include <memory>

// TODO Reduce duplicate code

namespace onert::ir::train
{

UseDefGenerator::UseDefGenerator(const TrainableGraph &tgraph)
  : _tgraph{tgraph}, _node_to_idx{}, _training_usedefs{}
{
  const auto order = _tgraph.topolSortOperations();
  for (const auto &index : order)
  {
    const auto &node = _tgraph.operation(index);
    assert(_node_to_idx.find(&node) == _node_to_idx.end());
    _node_to_idx[&node] = index;
  }

  // Check whether loss exists
  assert(std::any_of(order.begin(), order.end(),
                     [&](const auto &index) {
                       return _tgraph.operation(index).opcode() == ir::OpCode::Loss;
                     }) &&
         "Loss does not exist");
}

UseDefChains UseDefGenerator::operator()()
{
  const auto &graph = _tgraph.graph();
  assert(ir::verifier::EdgeChecker().verify(graph));

  _training_usedefs.clear();
  graph.operands().iterate([&](const ir::OperandIndex &idx, const ir::Operand &operand) {
    // Initialize as emtpy UseDefChain
    const auto empty_usedef_chain = UseDefChain{operand};
    _training_usedefs.emplace(TrainingOperandIndex{idx, true}, empty_usedef_chain);
    _training_usedefs.emplace(TrainingOperandIndex{idx, false}, empty_usedef_chain);
  });

  initForForwardingNodes();

  initForBackwardingNodes();

  return _training_usedefs;
}

void UseDefGenerator::visit(const train::operation::BinaryArithmetic &node)
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

  for (const auto &in_index : node.getUsedInputSet())
  {
    // Insert use of forwarding inputs
    const auto in_forwarding_index = TrainingOperandIndex{in_index, true};
    insertUse(in_forwarding_index, backwarding_op_index);

    // Set def of backwarding(backprop) inputs
    const auto outgoing_index = TrainingOperandIndex{in_index, false};
    insertBackPropDef(outgoing_index, backwarding_op_index);
  }
}

void UseDefGenerator::visit(const train::operation::Conv2D &node)
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

  // Insert use of forwarding output
  if (node.param().activation != ir::Activation::NONE)
  {
    const auto &out_index = node.getOutputs().at(0);
    const auto out_forwarding_index = TrainingOperandIndex{out_index, true};
    insertUse(out_forwarding_index, backwarding_op_index);
  }

  // Set def of backwarding inputs
  const auto outgoing_index = TrainingOperandIndex{in_index, false};
  insertBackPropDef(outgoing_index, backwarding_op_index);

  const auto weights_gradient_index = TrainingOperandIndex{weights_index, false};
  insertDef(weights_gradient_index, backwarding_op_index);

  const auto &bias_index = node.getInputs().at(ir::operation::Conv2D::Input::BIAS);
  if (bias_index.valid())
  {
    const auto bias_gradient_index = TrainingOperandIndex{bias_index, false};
    insertDef(bias_gradient_index, backwarding_op_index);
  }
}

void UseDefGenerator::visit(const train::operation::DepthwiseConv2D &node)
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

  // Insert uses of forwarding output
  if (node.param().activation != ir::Activation::NONE)
  {
    const auto &out_index = node.getOutputs().at(0);
    const auto out_forwarding_index = TrainingOperandIndex{out_index, true};
    insertUse(out_forwarding_index, backwarding_op_index);
  }

  // Set def of backwarding inputs
  const auto outgoing_index = TrainingOperandIndex{in_index, false};
  insertBackPropDef(outgoing_index, backwarding_op_index);

  const auto weights_gradient_index = TrainingOperandIndex{weights_index, false};
  insertDef(weights_gradient_index, backwarding_op_index);

  const auto &bias_index = node.getInputs().at(ir::operation::Conv2D::Input::BIAS);
  if (bias_index.valid())
  {
    const auto bias_gradient_index = TrainingOperandIndex{bias_index, false};
    insertDef(bias_gradient_index, backwarding_op_index);
  }
}

void UseDefGenerator::visit(const train::operation::ElementwiseActivation &node)
{
  if (node.param().op_type != operation::ElementwiseActivation::Type::RELU)
  {
    throw std::runtime_error{"UseDefGenerator: Not yet supported activation type"};
  }
  assert(_node_to_idx.find(&node) != _node_to_idx.end());
  const auto &op_index = _node_to_idx.at(&node);
  const auto backwarding_op_index = TrainingOperationIndex{op_index, false};

  // Insert use of forwarding output
  const auto &out_index = node.getOutputs().at(0);
  const auto out_forwarding_index = TrainingOperandIndex{out_index, true};
  insertUse(out_forwarding_index, backwarding_op_index);

  // Set def of backwarding(backprop) inputs
  for (const auto &in_index : node.getUsedInputSet())
  {
    const auto outgoing_index = TrainingOperandIndex{in_index, false};
    insertBackPropDef(outgoing_index, backwarding_op_index);
  }
}

void UseDefGenerator::visit(const train::operation::FullyConnected &node)
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

  // Insert uses of forwarding output
  if (node.param().activation != ir::Activation::NONE)
  {
    const auto &out_index = node.getOutputs().at(0);
    const auto out_forwarding_index = TrainingOperandIndex{out_index, true};
    insertUse(out_forwarding_index, backwarding_op_index);
  }

  // Set def of backwarding inputs
  const auto outgoing_index = TrainingOperandIndex{in_index, false};
  insertBackPropDef(outgoing_index, backwarding_op_index);

  const auto weights_gradient_index = TrainingOperandIndex{weights_index, false};
  insertDef(weights_gradient_index, backwarding_op_index);

  const auto &bias_index = node.getInputs().at(ir::operation::Conv2D::Input::BIAS);
  if (bias_index.valid())
  {
    const auto bias_gradient_index = TrainingOperandIndex{bias_index, false};
    insertDef(bias_gradient_index, backwarding_op_index);
  }
}

void UseDefGenerator::visit(const train::operation::Loss &node)
{
  assert(_node_to_idx.find(&node) != _node_to_idx.end());
  const auto &op_index = _node_to_idx.at(&node);
  const auto backwarding_op_index = TrainingOperationIndex{op_index, false};

  for (const auto &in_index : node.getUsedInputSet())
  {
    // Insert use of forwarding inputs
    const auto in_forwarding_index = TrainingOperandIndex{in_index, true};
    insertUse(in_forwarding_index, backwarding_op_index);
  }

  // Set def of backwarding(backprop) y_pred
  const auto &y_pred_index = node.getInputs().at(train::operation::Loss::Input::Y_PRED);
  assert(!_tgraph.operands().at(y_pred_index).isConstant());
  const auto y_pred_outgoing_index = TrainingOperandIndex{y_pred_index, false};
  insertBackPropDef(y_pred_outgoing_index, backwarding_op_index);

  // Set def of backwarding(backprop) y_true
  const auto &y_true_index = node.getInputs().at(train::operation::Loss::Input::Y_TRUE);
  assert(!_tgraph.operands().at(y_true_index).isConstant());
  const auto y_true_outgoing_index = TrainingOperandIndex{y_true_index, false};
  insertBackPropDef(y_true_outgoing_index, backwarding_op_index);

  // Remove use of backwarding output
  const auto &out_index = node.getOutputs().at(0);
  const auto incoming_index = TrainingOperandIndex{out_index, false};
  auto &usedef_chain = _training_usedefs.at(incoming_index);
  usedef_chain.removeTrainingUse(backwarding_op_index);
}

void UseDefGenerator::visit(const train::operation::Pad &node)
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
  const auto incoming_index = TrainingOperandIndex{out_index, false};
  insertUse(incoming_index, backwarding_op_index);

  // Set def of backwarding(backprop) input
  const auto &in_index = node.getInputs().at(train::operation::Pad::Input::INPUT);
  const auto outgoing_index = TrainingOperandIndex{in_index, false};
  insertBackPropDef(outgoing_index, backwarding_op_index);
}

void UseDefGenerator::visit(const train::operation::Pool2D &node)
{
  if (node.param().op_type != ir::operation::Pool2D::PoolType::MAX &&
      node.param().op_type != ir::operation::Pool2D::PoolType::AVG)
  {
    throw std::runtime_error{"UseDefGenerator: Not yet supported pool type"};
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
  const auto incoming_index = TrainingOperandIndex{out_index, false};
  insertUse(incoming_index, backwarding_op_index);

  // Set def of backwarding(backprop) input
  const auto &in_index = node.getInputs().at(train::operation::Pool2D::Input::INPUT);
  const auto outgoing_index = TrainingOperandIndex{in_index, false};
  insertBackPropDef(outgoing_index, backwarding_op_index);
}

void UseDefGenerator::visit(const train::operation::Reduce &node)
{
  if (node.param().reduce_type != ir::operation::Reduce::ReduceType::MEAN)
  {
    throw std::runtime_error{"UseDefGenerator: Not yet supported reduce type"};
  }

  assert(_node_to_idx.find(&node) != _node_to_idx.end());
  const auto &op_index = _node_to_idx.at(&node);
  const auto backwarding_op_index = TrainingOperationIndex{op_index, false};

  // Insert use of backwarding(backprop) output
  const auto &out_index = node.getOutputs().at(0);
  const auto incoming_index = TrainingOperandIndex{out_index, false};
  insertUse(incoming_index, backwarding_op_index);

  // Set def of backwarding(backprop) input
  const auto &in_index = node.getInputs().at(train::operation::Reduce::Input::INPUT);
  const auto outgoing_index = TrainingOperandIndex{in_index, false};
  insertBackPropDef(outgoing_index, backwarding_op_index);
}

void UseDefGenerator::visit(const train::operation::Reshape &node)
{
  assert(_node_to_idx.find(&node) != _node_to_idx.end());
  const auto &op_index = _node_to_idx.at(&node);
  const auto backwarding_op_index = TrainingOperationIndex{op_index, false};

  // Insert use of backwarding(backprop) output
  const auto &out_index = node.getOutputs().at(0);
  const auto incoming_index = TrainingOperandIndex{out_index, false};
  insertUse(incoming_index, backwarding_op_index);

  // Set def of backwarding(backprop) input
  const auto &in_index = node.getInputs().at(train::operation::Reduce::Input::INPUT);
  const auto outgoing_index = TrainingOperandIndex{in_index, false};
  insertBackPropDef(outgoing_index, backwarding_op_index);
}

void UseDefGenerator::visit(const train::operation::Softmax &node)
{
  assert(_node_to_idx.find(&node) != _node_to_idx.end());
  const auto &op_index = _node_to_idx.at(&node);
  const auto backwarding_op_index = TrainingOperationIndex{op_index, false};

  // Insert uses of forwarding output
  const auto &out_index = node.getOutputs().at(0);
  const auto out_forwarding_index = TrainingOperandIndex{out_index, true};
  insertUse(out_forwarding_index, backwarding_op_index);

  // Insert use of backwarding(backprop) output
  const auto incoming_index = TrainingOperandIndex{out_index, false};
  insertUse(incoming_index, backwarding_op_index);

  // Set def of backwarding(backprop) input
  const auto &in_index = node.getInputs().at(train::operation::Reduce::Input::INPUT);
  const auto outgoing_index = TrainingOperandIndex{in_index, false};
  insertBackPropDef(outgoing_index, backwarding_op_index);
}

void UseDefGenerator::insertUse(const TrainingOperandIndex &operand_index,
                                const TrainingOperationIndex &op_index)
{
  assert(_training_usedefs.find(operand_index) != _training_usedefs.end());
  auto &usedef_chain = _training_usedefs.at(operand_index);
  usedef_chain.insertTrainingUse(op_index);
}

void UseDefGenerator::insertDef(const TrainingOperandIndex &operand_index,
                                const TrainingOperationIndex &op_index)
{
  assert(operand_index.valid());

  assert(_training_usedefs.find(operand_index) != _training_usedefs.end());
  auto &usedef_chain = _training_usedefs.at(operand_index);
  usedef_chain.insertTrainingDef(op_index);
}

void UseDefGenerator::insertBackPropDef(const TrainingOperandIndex &operand_index,
                                        const TrainingOperationIndex &op_index)
{
  // NOTE There is no need to set def of constant backwarding(backprop) inputs
  //      because it won't be back-propagated.
  if (!_tgraph.operands().at(operand_index.index()).isConstant())
  {
    insertDef(operand_index, op_index);
  }
}

void UseDefGenerator::initForForwardingNodes()
{
  // Initialize training def-uses of forwarding operands for only forwarding nodes
  // (i.e. forwarding nodes that do not have any backwarding node)
  _tgraph.operands().iterate([&](const ir::OperandIndex &idx, const ir::Operand &operand) {
    // Append forwarding def-uses as it is
    const bool is_forward = true;
    const auto forwarding_operand_index = TrainingOperandIndex{idx, is_forward};

    const auto def = operand.getDef();
    if (def.valid())
    {
      insertDef(forwarding_operand_index, TrainingOperationIndex{def, is_forward});
      auto &usedef_chain = _training_usedefs.at(forwarding_operand_index);
      usedef_chain.insertTrainingDef(TrainingOperationIndex{def, is_forward});
    }

    assert(_training_usedefs.at(forwarding_operand_index).getTrainingUses().size() == 0);
    const auto uses(operand.getUses());
    for (const auto &use : uses)
      insertUse(forwarding_operand_index, TrainingOperationIndex{use, is_forward});
  });
}

void UseDefGenerator::initForBackwardingNodes()
{
  const auto backward_order = _tgraph.essentialBackwardOrder();
  // Initialize training uses of forwarding operands and def-uses of backwarding operands for
  // backwarding nodes (i.e. backwarding nodes that do not have any forwarding node)
  for (const auto &op_index : backward_order)
  {
    const auto &node = _tgraph.operation(op_index);

    // Insert use of backwarding operands(only output)
    {
      if (node.getOutputs().size() > 1)
        throw std::runtime_error(
          "UseDefGenerator does not support multiple outputs of training operation");

      const auto &output = node.getOutputs().at(0);
      const auto backwarding_op_index = TrainingOperationIndex{op_index, false};
      const auto incoming_index = TrainingOperandIndex{output, false};
      insertUse(incoming_index, backwarding_op_index);
    }

    // Insert uses of forwarding operands and insert defs of backwarding operands
    node.accept(*this);
  }
}

} // namespace onert::ir::train
