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

#include "compiler/StaticShapeInferer.h"
#include "util/ShapeInference.h"
#include "util/logging.h"

#include <misc/polymorphic_downcast.h>

#include <sstream>
#include <stdexcept>

namespace onert
{
namespace compiler
{
void OperandObserver::updateShapes(const std::vector<ir::OperandInfo> &changed_operands_info,
                                   bool unpredictable)
{
  assert(changed_operands_info.size() == _operands.size());
  for (size_t i = 0; i < changed_operands_info.size(); ++i)
  {
    const auto &changed_operand_info = changed_operands_info.at(i);
    auto &operand = _operands.at(i);
    // assert(changed_operand_info.typeInfo() == operand->typeInfo());
    // assert(changed_operand_info.typeInfo() == operand->typeInfo());
    // This error check may by replaced by an assertion if this function is called after the
    // validation of models are completed.
    if (changed_operand_info.typeInfo() != operand->typeInfo())
    {
      throw std::runtime_error("OperandObserver: The types of operands are mismatched");
    }
    if (!operand->info().isConstant() && (changed_operand_info.isDynamic() || unpredictable))
    {
      operand->info().setDynamic();
    }
    else
    {
      const auto &new_shape = changed_operands_info.at(i).shape();
      operand->info().shape(new_shape);
    }
  }
}

void StaticShapeInferer::infer()
{
  for (const auto &op_idx : _lowered_subg->graph().topolSortOperations())
  {
    const auto &op = _lowered_subg->graph().operations().at(op_idx);
    bool has_dynamic_tensor = false;
    const auto opcode = op.opcode();
    // IF: requires shape inference for then, else
    // While: requires shape inference for condition, body
    if (opcode == ir::OpCode::If || opcode == ir::OpCode::While)
    {
      op.accept(*this);
    }
    else
    {
      has_dynamic_tensor = checkDynamicInput(op);
      if (has_dynamic_tensor)
      {
        setDynamicOutput(op);
      }
      else
      {
        op.accept(*this);
      }
    }
    has_dynamic_tensor = has_dynamic_tensor || checkDynamicOutput(op);
    _lowered_subg->setHasDynamicTensor(op_idx, has_dynamic_tensor);
  }

  if (_controlflow_output_observer != nullptr)
  {
    // re-sizing output shapes of the controflow operation branching to this subgraph
    std::vector<ir::OperandInfo> outputs_info;
    const auto &graph = _lowered_subg->graph();
    const auto &outputs = graph.getOutputs();
    for (size_t i = 0; i < outputs.size(); ++i)
    {
      const auto &operand_info = graph.operands().at(outputs.at(i)).info();
      outputs_info.emplace_back(operand_info);
    }
    _controlflow_output_observer->updateShapes(outputs_info);
  }
}

bool StaticShapeInferer::checkDynamicInput(const ir::IOperation &op)
{
  const auto &operands = _lowered_subg->graph().operands();
  for (auto &&input_idx : op.getInputs() | ir::Remove::UNDEFINED | ir::Remove::DUPLICATED)
  {
    if (operands.at(input_idx).info().isDynamic())
    {
      return true;
    }
  }

  return false;
}

bool StaticShapeInferer::checkDynamicOutput(const ir::IOperation &op)
{
  auto &operands = _lowered_subg->graph().operands();
  for (auto &&output_idx : op.getOutputs() | ir::Remove::UNDEFINED)
  {
    if (operands.at(output_idx).info().isDynamic())
    {
      return true;
    }
  }
  return false;
}

void StaticShapeInferer::setDynamicOutput(const ir::IOperation &op)
{
  auto &operands = _lowered_subg->graph().operands();
  for (auto &&output_idx : op.getOutputs() | ir::Remove::UNDEFINED)
  {
    operands.at(output_idx).info().setDynamic();
  }
}

void StaticShapeInferer::handleBinaryArithmeticOp(const ir::Operation &op,
                                                  const ir::OperandIndex lhs_idx,
                                                  const ir::OperandIndex rhs_idx)
{
  auto &operands = _lowered_subg->graph().operands();
  const auto &lhs = operands.at(lhs_idx);
  const auto &rhs = operands.at(rhs_idx);

  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = operands.at(output_idx);

  // re-sizing output shape
  ir::Shape new_shape = shape_inference::inferEltwiseShape(lhs.info().shape(), rhs.info().shape());
  output.info().shape(new_shape);
}

void StaticShapeInferer::handleSimpleUnaryOp(const ir::Operation &op,
                                             const ir::OperandIndex input_idx)
{
  auto &operands = _lowered_subg->graph().operands();
  const auto &input = operands.at(input_idx);

  // get mutable output operand
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = operands.at(output_idx);

  // re-sizing output shape
  ir::Shape new_shape = input.info().shape();
  output.info().shape(new_shape);
}

void StaticShapeInferer::dump()
{
  auto get_shape_str = [](const ir::Shape &shape) {
    std::stringstream sstream;
    sstream << "shape : {";
    for (int i = 0; i < shape.rank(); i++)
    {
      if (i == 0)
        sstream << shape.dim(i);
      else
        sstream << " " << shape.dim(i);
    }
    sstream << "}";
    return sstream.str();
  };

  _lowered_subg->graph().operands().iterate(
    [&](const ir::OperandIndex &ind, const ir::Operand &operand) {
      VERBOSE(StaticShapeInferer) << "  " << ind << ", "
                                  << (operand.info().isDynamic() ? "Dynamic" : "Static") << ", "
                                  << get_shape_str(operand.info().shape()) << std::endl;
    });
}

std::unordered_map<ir::SubgraphIndex, std::unique_ptr<StaticShapeInferer>>
StaticShapeInferer::createStaticShapeInferers(
  const std::unordered_map<ir::SubgraphIndex, ILoweredGraph *> &lowered_subgs)
{
  // Allocate StaticShapeInferer per each subgraph
  std::unordered_map<ir::SubgraphIndex, std::unique_ptr<StaticShapeInferer>> inferers;
  for (auto &&pair : lowered_subgs)
  {
    const auto &subg_index = pair.first;
    auto &lowered_subg = pair.second;
    inferers[subg_index] = std::make_unique<StaticShapeInferer>(lowered_subg);
  }

  // Append observers in all StaticShapeInferers
  for (auto &&pair : lowered_subgs)
  {
    const auto &subg_index = pair.first;
    auto &lowered_subg = pair.second;

    // TODO: Change this iteration for all to controlflow iteration
    lowered_subg->graph().operations().iterate(
      [&](const ir::OperationIndex &, const ir::IOperation &op) {
        // A Function to append child inferers. These make it possible for a StaticShapeInferer to
        // call StaticShapeInferes of child subgraphs recursively
        auto appendChildInferer = [&](const ir::SubgraphIndex &child_subg_idx) {
          auto *child_inferer = inferers.at(child_subg_idx).get();
          inferers.at(subg_index)->appendChildInferer(child_subg_idx, child_inferer);
        };

        // A Function to appaend subg input observers. This makes it possible for a
        // StaticShapeInferer to update inputs of child subgraphs
        auto appendSubgraphInputObserver = [&](const ir::SubgraphIndex &child_subg_idx) {
          std::vector<ir::Operand *> child_subg_inputs;
          auto &child_subg = lowered_subgs.at(child_subg_idx)->graph();
          for (const auto &input_idx : child_subg.getInputs())
          {
            auto operand_ptr = child_subg.operands().getRawPtr(input_idx);
            child_subg_inputs.emplace_back(operand_ptr);
          }
          inferers.at(subg_index)
            ->appendSubgInputObserver(child_subg_idx,
                                      std::make_unique<OperandObserver>(child_subg_inputs));
        };

        // A Function to set controlflow output observers. This makes it possible for a
        // StaticShapeInferer to update outputs of parent controlflow opeerations
        auto setControlFlowOutputObserver = [&](const ir::SubgraphIndex &child_subg_idx) {
          std::vector<ir::Operand *> cf_outputs;
          auto &subg = lowered_subg->graph();
          for (const auto &output_idx : op.getOutputs())
          {
            auto operand_ptr = subg.operands().getRawPtr(output_idx);
            cf_outputs.emplace_back(operand_ptr);
          }
          inferers.at(child_subg_idx)
            ->setControlflowOutputObserver(std::make_unique<OperandObserver>(cf_outputs));
        };

        // Append Observers in a StaticShapeInferer
        if (op.opcode() == ir::OpCode::If)
        {
          // TODO Remove dynamic_cast
          // An virtual base class cannot be downcasted by static_cast
          try
          {
            const auto &if_op = dynamic_cast<const ir::operation::If &>(op);

            appendChildInferer(if_op.param().then_subg_index);
            appendChildInferer(if_op.param().else_subg_index);

            appendSubgraphInputObserver(if_op.param().then_subg_index);
            appendSubgraphInputObserver(if_op.param().else_subg_index);

            setControlFlowOutputObserver(if_op.param().then_subg_index);
          }
          catch (const std::bad_cast &)
          {
            throw std::runtime_error("StaticShapeInferer: Invalid If operation");
          }
        }
        else if (op.opcode() == ir::OpCode::While)
        {
          // TODO Remove dynamic_cast
          try
          {
            const auto &while_op = dynamic_cast<const ir::operation::While &>(op);

            appendChildInferer(while_op.param().cond_subg_index);
            appendChildInferer(while_op.param().body_subg_index);

            appendSubgraphInputObserver(while_op.param().cond_subg_index);
            appendSubgraphInputObserver(while_op.param().body_subg_index);

            setControlFlowOutputObserver(while_op.param().body_subg_index);
          }
          catch (const std::bad_cast &)
          {
            throw std::runtime_error("StaticShapeInferer: Invalid While operation");
          }
        }
      });
  }

  return inferers;
}

void StaticShapeInferer::visit(const ir::operation::ArgMinMax &op)
{
  auto &operands = _lowered_subg->graph().operands();

  const auto input_idx{op.getInputs().at(ir::operation::ArgMinMax::Input::INPUT)};
  const auto &input = operands.at(input_idx);

  const auto axis_idx{op.getInputs().at(ir::operation::ArgMinMax::Input::AXIS)};
  const auto &axis = operands.at(axis_idx);

  // get mutable output operand
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = operands.at(output_idx);

  if (!axis.isConstant())
  {
    output.info().setDynamic();
    return;
  }

  const auto rank = input.info().shape().rank();
  auto axis_value = axis.asScalar<int32_t>();
  axis_value = axis_value < 0 ? axis_value + rank : axis_value;

  // re-sizing output shape
  ir::Shape new_shape =
    shape_inference::inferArgMinMaxShape(input.info().shape(), axis_value, rank);
  output.info().shape(new_shape);
}

void StaticShapeInferer::visit(const ir::operation::BatchMatMul &op)
{
  auto &operands = _lowered_subg->graph().operands();

  const auto lhs_index = op.getInputs().at(ir::operation::BatchMatMul::Input::LHS);
  const auto rhs_index = op.getInputs().at(ir::operation::BatchMatMul::Input::RHS);
  const auto output_index = op.getOutputs().at(0);
  const auto &lhs = operands.at(lhs_index);
  const auto &rhs = operands.at(rhs_index);
  auto &output = operands.at(output_index);
  auto new_shape = shape_inference::inferBatchMatMulShape(lhs.shape(), rhs.shape(), op.param());
  output.info().shape(new_shape);
}

void StaticShapeInferer::visit(const ir::operation::BCQFullyConnected &op)
{
  auto &operands = _lowered_subg->graph().operands();

  const auto input_idx{op.getInputs().at(ir::operation::BCQFullyConnected::Input::INPUT)};
  const auto &input = operands.at(input_idx);

  const auto cluster_idx{
    op.getInputs().at(ir::operation::BCQFullyConnected::Input::WEIGHTS_CLUSTERS)};
  const auto &cluster = operands.at(cluster_idx);

  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = operands.at(output_idx);

  auto cluster_buf = reinterpret_cast<const int32_t *>(cluster.data()->base());
  assert(cluster_buf);

  // re-sizing output shape
  ir::Shape new_shape = shape_inference::inferBCQFullyConnectedShape(
    input.info().shape(), cluster.info().shape(), cluster_buf);
  output.info().shape(new_shape);
}

void StaticShapeInferer::visit(const ir::operation::BCQGather &op)
{
  auto &operands = _lowered_subg->graph().operands();

  const auto indices_idx{op.getInputs().at(ir::operation::BCQGather::Input::INDICES)};
  const auto &indices = operands.at(indices_idx);

  const auto input_binary_idx{op.getInputs().at(ir::operation::BCQGather::Input::INPUT_BINARY)};
  const auto &input_binary = operands.at(input_binary_idx);

  const auto cluster_idx{op.getInputs().at(ir::operation::BCQGather::Input::INPUT_CLUSTERS)};
  const auto &cluster = operands.at(cluster_idx);

  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = operands.at(output_idx);

  auto cluster_buf = reinterpret_cast<const int32_t *>(cluster.data()->base());
  assert(cluster_buf);

  auto rank = input_binary.shape().rank();

  // re-sizing output shape
  ir::Shape new_shape = shape_inference::inferBCQGatherShape(
    indices.info().shape(), cluster.info().shape(), cluster_buf, rank, op.param());

  output.info().shape(new_shape);
}

void StaticShapeInferer::visit(const ir::operation::BinaryArithmetic &op)
{
  handleBinaryArithmeticOp(op, op.getInputs().at(ir::operation::BinaryArithmetic::Input::LHS),
                           op.getInputs().at(ir::operation::BinaryArithmetic::Input::RHS));
}

void StaticShapeInferer::visit(const ir::operation::BroadcastTo &op)
{
  // get mutable output operand
  auto &operands = _lowered_subg->graph().operands();
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = operands.at(output_idx);

  const auto shape_idx{op.getInputs().at(ir::operation::BroadcastTo::Input::SHAPE)};
  const auto &shape = operands.at(shape_idx);

  if (!shape.isConstant())
  {
    output.info().setDynamic();
    return;
  }

  // assert(shape.typeInfo().type() == ir::DataType::INT32);
  auto shape_buffer = reinterpret_cast<const int32_t *>(shape.data()->base());

  // re-sizing output shape
  ir::Shape new_shape = shape_inference::inferBroadcastToShape(shape.info().shape(), shape_buffer);
  output.info().shape(new_shape);
}

void StaticShapeInferer::visit(const ir::operation::Comparison &op)
{
  handleBinaryArithmeticOp(op, op.getInputs().at(ir::operation::Comparison::Input::INPUT0),
                           op.getInputs().at(ir::operation::Comparison::Input::INPUT1));
}

void StaticShapeInferer::visit(const ir::operation::Concat &op)
{
  auto &operands = _lowered_subg->graph().operands();

  const auto input_count = op.getInputs().size();

  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = operands.at(output_idx);

  shape_inference::Shapes input_shapes;
  for (uint32_t i = 0; i < input_count; i++)
  {
    const auto input_idx{op.getInputs().at(i)};
    const auto &input = operands.at(input_idx);
    input_shapes.emplace_back(input.shape());
  }

  ir::Shape out_shape = shape_inference::inferConcatShape(input_shapes, op.param());

  // re-sizing output shape
  output.info().shape(out_shape);
}

void StaticShapeInferer::visit(const ir::operation::Conv2D &op)
{
  auto &operands = _lowered_subg->graph().operands();

  const auto input_idx{op.getInputs().at(ir::operation::Conv2D::Input::INPUT)};
  const auto &input = operands.at(input_idx);
  const auto ker_idx{op.getInputs().at(ir::operation::Conv2D::Input::KERNEL)};
  const auto &ker = operands.at(ker_idx);
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = operands.at(output_idx);

  // re-sizing output shape
  ir::Shape new_shape =
    shape_inference::inferConv2DShape(input.info().shape(), ker.info().shape(), op.param());
  output.info().shape(new_shape);
}

void StaticShapeInferer::visit(const ir::operation::DepthwiseConv2D &op)
{
  auto &operands = _lowered_subg->graph().operands();

  const auto input_idx{op.getInputs().at(ir::operation::DepthwiseConv2D::Input::INPUT)};
  const auto &input = operands.at(input_idx);
  const auto ker_idx{op.getInputs().at(ir::operation::DepthwiseConv2D::Input::KERNEL)};
  const auto &ker = operands.at(ker_idx);
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = operands.at(output_idx);

  // re-sizing output shape
  ir::Shape new_shape = shape_inference::inferDepthwiseConv2DShape(input.info().shape(),
                                                                   ker.info().shape(), op.param());
  output.info().shape(new_shape);
}

void StaticShapeInferer::visit(const ir::operation::ElementwiseActivation &op)
{
  handleSimpleUnaryOp(op, op.getInputs().at(ir::operation::ElementwiseActivation::Input::INPUT));
}

void StaticShapeInferer::visit(const ir::operation::ElementwiseBinary &op)
{
  handleBinaryArithmeticOp(op, op.getInputs().at(ir::operation::ElementwiseBinary::Input::LHS),
                           op.getInputs().at(ir::operation::ElementwiseBinary::Input::RHS));
}

void StaticShapeInferer::visit(const ir::operation::ElementwiseUnary &op)
{
  handleSimpleUnaryOp(op, op.getInputs().at(ir::operation::ElementwiseUnary::Input::INPUT));
}

void StaticShapeInferer::visit(const ir::operation::ExpandDims &op)
{
  auto &operands = _lowered_subg->graph().operands();

  const auto input_idx{op.getInputs().at(ir::operation::ExpandDims::Input::INPUT)};
  const auto &input = operands.at(input_idx);
  const auto axis_idx{op.getInputs().at(ir::operation::ExpandDims::Input::AXIS)};
  const auto &axis = operands.at(axis_idx);
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = operands.at(output_idx);

  if (!axis.isConstant())
  {
    output.info().setDynamic();
    return;
  }

  // even when axis is constant, output shape should be recalculated since user might call
  // nnfw_set_input_tensorinfo(input, some_new_shape)
  auto axis_type = axis.typeInfo().type();
  assert(axis_type == ir::DataType::INT32 || axis_type == ir::DataType::INT64);

  assert(axis.data()->base());
  int32_t axis_value =
    (axis_type == ir::DataType::INT32)
      ? reinterpret_cast<const int32_t *>(axis.data()->base())[0]
      : static_cast<int32_t>(reinterpret_cast<const int64_t *>(axis.data()->base())[0]);

  // re-sizing output shape
  ir::Shape new_shape = shape_inference::inferExpandDimsShape(input.info().shape(), axis_value);
  output.info().shape(new_shape);
}

void StaticShapeInferer::visit(const ir::operation::Fill &op)
{
  auto &operands = _lowered_subg->graph().operands();

  const auto shape_idx{op.getInputs().at(ir::operation::Fill::Input::SHAPE)};
  const auto &shape = operands.at(shape_idx);
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = operands.at(output_idx);

  if (!shape.isConstant())
  {
    output.info().setDynamic();
    return;
  }

  const auto dims_type = shape.typeInfo().type();
  assert(dims_type == ir::DataType::INT32 || dims_type == ir::DataType::INT64);

  auto dims_buf = shape.data()->base();
  assert(dims_buf);

  const auto &dims_shape = shape.info().shape();
  const auto &new_shape = ((dims_type == ir::DataType::INT32)
                             ? shape_inference::inferFillShape<int32_t>(
                                 dims_shape, reinterpret_cast<const int32_t *>(dims_buf))
                             : shape_inference::inferFillShape<int64_t>(
                                 dims_shape, reinterpret_cast<const int64_t *>(dims_buf)));

  output.info().shape(new_shape);
}

void StaticShapeInferer::visit(const ir::operation::FullyConnected &op)
{
  auto &operands = _lowered_subg->graph().operands();

  const auto input_idx{op.getInputs().at(ir::operation::FullyConnected::Input::INPUT)};
  const auto &input = operands.at(input_idx);

  const auto ker_idx{op.getInputs().at(ir::operation::FullyConnected::Input::WEIGHT)};
  const auto &ker = operands.at(ker_idx);

  // get mutable output operand
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = operands.at(output_idx);
  // re-sizing output shape
  ir::Shape new_shape =
    shape_inference::inferFullyConnectedShape(input.info().shape(), ker.info().shape());
  output.info().shape(new_shape);
}

void StaticShapeInferer::visit(const ir::operation::FusedBatchNorm &op)
{
  handleSimpleUnaryOp(op, op.getInputs().at(ir::operation::FusedBatchNorm::Input::INPUT));
}

void StaticShapeInferer::visit(const ir::operation::Gather &op)
{
  auto &operands = _lowered_subg->graph().operands();

  const auto input_idx{op.getInputs().at(ir::operation::Gather::Input::INPUT)};
  const auto &input = operands.at(input_idx);

  // get mutable output operand
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = operands.at(output_idx);

  const auto indices_idx{op.getInputs().at(ir::operation::Gather::Input::INDICES)};
  const auto &indices = operands.at(indices_idx);
  const auto rank = input.info().shape().rank();
  const auto axis = ((op.param().axis < 0) ? rank + op.param().axis : op.param().axis);

  assert(0 <= axis && axis < rank);

  // re-sizing output shape
  ir::Shape new_shape =
    shape_inference::inferGatherShape(input.info().shape(), indices.info().shape(), axis, rank);
  output.info().shape(new_shape);
}

void StaticShapeInferer::visit(const ir::operation::If &op)
{
  // re-sizing input shapes of then/else subgraph
  const std::vector<ir::OperandIndex> inputs{op.getInputs().begin() + 1, op.getInputs().end()};

  std::vector<ir::OperandInfo> inputs_info;
  const auto &graph = _lowered_subg->graph();
  for (size_t i = 0; i < inputs.size(); ++i)
  {
    const auto &operand_info = graph.operands().at(inputs.at(i)).info();
    inputs_info.emplace_back(operand_info);
  }
  _subg_input_observers.at(op.param().then_subg_index)->updateShapes(inputs_info);
  _child_inferers.at(op.param().then_subg_index)->infer();

  _subg_input_observers.at(op.param().else_subg_index)->updateShapes(inputs_info);
  _child_inferers.at(op.param().else_subg_index)->infer();
}

void StaticShapeInferer::visit(const ir::operation::L2Normalization &op)
{
  handleSimpleUnaryOp(op, op.getInputs().at(ir::operation::L2Normalization::Input::INPUT));
}

void StaticShapeInferer::visit(const ir::operation::Loss &op)
{
  // TODO Consider SparseCategoricalCrossentropy case

  auto &operands = _lowered_subg->graph().operands();

  const auto input_index{op.getInputs().at(ir::operation::Loss::Input::Y_PRED)};
  auto &input = operands.at(input_index);

  const auto output_index{op.getOutputs().at(0)};
  auto &output = operands.at(output_index);

  ir::Shape new_shape = output.info().shape();
  new_shape.dim(0) = input.info().shape().dim(0);

  output.info().shape(new_shape);
}

void StaticShapeInferer::visit(const ir::operation::LSTM &op)
{
  auto &operands = _lowered_subg->graph().operands();

  const auto output_index{op.getOutputs().at(ir::operation::LSTM::Output::OUTPUT)};
  auto &output = operands.at(output_index);

  const auto output_state_out_index{
    op.getOutputs().at(ir::operation::LSTM::Output::OUTPUT_STATE_OUT)};

  const auto cell_state_out_index{op.getOutputs().at(ir::operation::LSTM::Output::CELL_STATE_OUT)};

  const auto scratch_buffer_index{op.getOutputs().at(ir::operation::LSTM::Output::SCRATCH_BUFFER)};

  if (output.info().isDynamic() ||
      (operands.exist(output_state_out_index) &&
       operands.at(output_state_out_index).info().isDynamic()) ||
      (operands.exist(cell_state_out_index) &&
       operands.at(cell_state_out_index).info().isDynamic()) ||
      (operands.exist(scratch_buffer_index) &&
       operands.at(scratch_buffer_index).info().isDynamic()))
    return;

  const auto input_index{op.getInputs().at(ir::operation::LSTM::Input::INPUT)};
  const auto &input = operands.at(input_index);

  const auto input_to_output_weights_index{
    op.getInputs().at(ir::operation::LSTM::Input::INPUT_TO_OUTPUT_WEIGHTS)};
  const auto &input_to_output_weights = operands.at(input_to_output_weights_index);

  const auto recurrent_to_output_weights_index{
    op.getInputs().at(ir::operation::LSTM::Input::RECURRENT_TO_OUTPUT_WEIGHTS)};
  const auto &recurrent_to_output_weights = operands.at(recurrent_to_output_weights_index);

  // re-sizing outputs
  const int n_batch = (input.shape().rank() == 3 && op.param().time_major) ? input.shape().dim(1)
                                                                           : input.shape().dim(0);
  const int n_cell = input_to_output_weights.shape().dim(0);
  const int n_output = recurrent_to_output_weights.shape().dim(1);
  if (input.shape().rank() == 3)
  {
    if (op.param().time_major)
      output.info().shape(ir::Shape{input.shape().dim(0), n_batch, n_output});
    else
      output.info().shape(ir::Shape{n_batch, input.shape().dim(1), n_output});
  }
  else
  {
    assert(input.shape().rank() == 2);
    output.info().shape(ir::Shape{n_batch, n_output});
  }

  if (operands.exist(output_state_out_index))
  {
    auto &output_state_out = operands.at(output_state_out_index);
    output_state_out.info().shape(ir::Shape{n_batch, n_output});
  }

  if (operands.exist(cell_state_out_index))
  {
    auto &cell_state_out = operands.at(cell_state_out_index);
    cell_state_out.info().shape(ir::Shape{n_batch, n_cell});
  }

  if (operands.exist(scratch_buffer_index))
  {
    auto &scratch_buffer = operands.at(scratch_buffer_index);

    const auto input_to_input_weights_index{
      op.getInputs().at(ir::operation::LSTM::Input::INPUT_TO_INPUT_WEIGHTS)};
    const auto recurrent_to_input_weights_index{
      op.getInputs().at(ir::operation::LSTM::Input::RECURRENT_TO_INPUT_WEIGHTS)};

    bool has_input_to_input_weights =
      operands.at(input_to_input_weights_index).shape().dim(0) != 0 &&
      operands.at(input_to_input_weights_index).shape().dim(1) != 0;
    bool has_recurrent_to_input_weights =
      operands.at(recurrent_to_input_weights_index).shape().dim(0) != 0 &&
      operands.at(recurrent_to_input_weights_index).shape().dim(1) != 0;

    // NOTE The cell_to_input_weights do not exist in non-peephole although regular LSTM(non-CIFG).
    // true: no CIFG
    // false: CIFG
    bool has_cifg_param = has_input_to_input_weights && has_recurrent_to_input_weights;
    if (has_cifg_param)
    {
      scratch_buffer.info().shape(ir::Shape{n_batch, n_cell * 4});
    }
    else
    {
      scratch_buffer.info().shape(ir::Shape{n_batch, n_cell * 3});
    }
  }
}

void StaticShapeInferer::visit(const ir::operation::MatrixBandPart &op)
{
  handleSimpleUnaryOp(op, op.getInputs().at(ir::operation::MatrixBandPart::Input::INPUT));
}

void StaticShapeInferer::visit(const ir::operation::OneHot &op)
{
  auto &operands = _lowered_subg->graph().operands();

  const auto indice_idx{op.getInputs().at(ir::operation::OneHot::Input::INDICES)};
  const auto &indice = operands.at(indice_idx);
  const auto depth_idx{op.getInputs().at(ir::operation::OneHot::Input::DEPTH)};
  const auto &depth = operands.at(depth_idx);

  const auto axis = op.param().axis;

  auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = operands.at(output_idx);

  if (!depth.isConstant())
  {
    output.info().setDynamic();
    return;
  }

  const auto *depth_buf = reinterpret_cast<const int32_t *>(depth.data()->base());
  assert(depth_buf);
  // re-sizing output shape
  ir::Shape new_shape = shape_inference::inferOnehotShape(indice.info().shape(), *depth_buf, axis);
  output.info().shape(new_shape);
}

void StaticShapeInferer::visit(const ir::operation::Pack &op)
{
  auto &operands = _lowered_subg->graph().operands();

  const auto input_idx{op.getInputs().at(0)};
  const auto &input = operands.at(input_idx);

  // get mutable output operand
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = operands.at(output_idx);

  const auto rank = input.shape().rank() + 1;
  const auto axis = ((op.param().axis < 0) ? rank + op.param().axis : op.param().axis);
  const auto num = op.param().num;

  assert(0 <= axis && axis < rank);

  // re-sizing output shape
  ir::Shape new_shape = shape_inference::inferPackShape(input.info().shape(), axis, rank, num);
  output.info().shape(new_shape);
}

void StaticShapeInferer::visit(const ir::operation::Pad &op)
{
  auto &operands = _lowered_subg->graph().operands();

  const auto input_idx{op.getInputs().at(ir::operation::Pad::Input::INPUT)};
  const auto &input = operands.at(input_idx);

  const auto pad_idx{op.getInputs().at(ir::operation::Pad::Input::PAD)};
  const auto &pad = operands.at(pad_idx);

  // get mutable output operand
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = operands.at(output_idx);

  // if pad is not constant, output also becomes dynamic
  if (!pad.isConstant())
  {
    output.info().setDynamic();
    return;
  }

  // re-sizing output shape
  const auto &new_shape = shape_inference::inferPadShape(
    input.shape(), reinterpret_cast<const int32_t *>(pad.data()->base()),
    pad.shape().num_elements());
  output.info().shape(new_shape);
}

void StaticShapeInferer::visit(const ir::operation::Permute &op)
{
  auto &operands = _lowered_subg->graph().operands();

  const auto input_idx{op.getInputs().at(0)};
  const auto &input = operands.at(input_idx);
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = operands.at(output_idx);

  // re-sizing output shape
  // Permute is a special operation that layouts of input/output may be different on backend
  // However, it is not applied here, so input/output have the same layout of frontend. Because
  // "ExecutorFactory" would convert shape of input/output accoding to the layouts when registering
  // operand info to "TensorBuilder" after calling "StaticShapeInferer"
  const auto &new_shape = input.info().shape();
  output.info().shape(new_shape);
}

void StaticShapeInferer::visit(const ir::operation::Pool2D &op)
{
  auto &operands = _lowered_subg->graph().operands();

  const auto layout = _lowered_subg->graph().layout();

  const auto input_idx{op.getInputs().at(ir::operation::Pool2D::Input::INPUT)};
  const auto &input = operands.at(input_idx);
  if (input.info().shape().rank() != 4)
  {
    throw std::runtime_error(op.name() + ": supports only 4D tensor as input");
  }

  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = operands.at(output_idx);

  ir::Shape new_shape = shape_inference::inferPoolShape(input.info().shape(), op.param(), layout);
  output.info().shape(new_shape);
}

void StaticShapeInferer::visit(const ir::operation::Pow &op)
{
  handleBinaryArithmeticOp(op, op.getInputs().at(ir::operation::Pow::Input::LHS),
                           op.getInputs().at(ir::operation::Pow::Input::RHS));
}

void StaticShapeInferer::visit(const ir::operation::Range &op)
{
  auto &operands = _lowered_subg->graph().operands();

  const auto start_idx{op.getInputs().at(ir::operation::Range::Input::START)};
  const auto limit_idx{op.getInputs().at(ir::operation::Range::Input::LIMIT)};
  const auto delta_idx{op.getInputs().at(ir::operation::Range::Input::DELTA)};
  const auto &start_op = operands.at(start_idx);
  const auto &limit_op = operands.at(limit_idx);
  const auto &delta_op = operands.at(delta_idx);

  // get mutable output operand
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = operands.at(output_idx);

  ir::Shape new_shape;
  if (start_op.isConstant() && limit_op.isConstant() && delta_op.isConstant())
  {
    assert(start_op.typeInfo().type() == limit_op.typeInfo().type() &&
           start_op.typeInfo().type() == delta_op.typeInfo().type());
    if (output.typeInfo().type() == ir::DataType::FLOAT32)
    {
      new_shape = shape_inference::inferRangeShape<float>(
        start_op.asScalar<float>(), limit_op.asScalar<float>(), delta_op.asScalar<float>());
    }
    else if (output.typeInfo().type() == ir::DataType::INT32)
    {
      new_shape = shape_inference::inferRangeShape<int32_t>(
        start_op.asScalar<int32_t>(), limit_op.asScalar<int32_t>(), delta_op.asScalar<int32_t>());
    }
    assert(output.shape() == new_shape);
  }
  else
  {
    output.info().setDynamic();
  }
}

void StaticShapeInferer::visit(const ir::operation::Reduce &op)
{
  auto &operands = _lowered_subg->graph().operands();

  const auto input_idx{op.getInputs().at(ir::operation::Reduce::Input::INPUT)};
  const auto &input = operands.at(input_idx);

  const auto axes_idx{op.getInputs().at(ir::operation::Reduce::Input::AXES)};
  const auto &axes = operands.at(axes_idx);

  // get mutable output operand
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = operands.at(output_idx);

  std::vector<int32_t> axes_vec;
  for (size_t i = 0; i < axes.shape().num_elements(); ++i)
  {
    switch (axes.typeInfo().type())
    {
      case ir::DataType::INT32:
      {
        axes_vec.emplace_back(reinterpret_cast<const int32_t *>(axes.data()->base())[i]);
        break;
      }
      case ir::DataType::INT64:
      {
        axes_vec.emplace_back(reinterpret_cast<const int64_t *>(axes.data()->base())[i]);
        break;
      }
      default:
        throw std::runtime_error("StaticShapeInferer " + op.name() + ": Not supported data type");
        break;
    }
  }
  const auto keep_dims = op.param().keep_dims;

  // re-sizing output shape
  ir::Shape new_shape =
    shape_inference::inferReduceShape(input.info().shape(), axes_vec, keep_dims);
  output.info().shape(new_shape);
}

void StaticShapeInferer::visit(const ir::operation::Reshape &op)
{
  auto &operands = _lowered_subg->graph().operands();

  const auto input_idx{op.getInputs().at(ir::operation::Reshape::Input::INPUT)};
  const auto &input = operands.at(input_idx);

  // get mutable output operand
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = operands.at(output_idx);

  // New shape is given by second input tensor
  if (op.getInputs().size() == 2)
  {
    // Let's check the second input
    const auto shape_idx{op.getInputs().at(ir::operation::Reshape::Input::SHAPE)};
    const auto &shape = operands.at(shape_idx);

    if (shape.isConstant())
    {
      const auto *shape_buf = reinterpret_cast<const int32_t *>(shape.data()->base());
      assert(shape_buf);

      ir::Shape new_shape =
        shape_inference::inferReshapeShape(input.shape(), shape_buf, shape.shape().num_elements());

      // if shape is from Const, TFLC put the shape of output into tensor
      if (new_shape != output.shape())
      {
        // change on output shape
        output.info().shape(new_shape);
      }
    }
    else
    {
      // if shape is NOT Const, set output shape to be dynamic_
      output.info().setDynamic();
    }
  }
  // New shape is given by option
  else if (op.param().new_shape.size() != 0)
  {
    // Let's check the new_shape option
    auto shape = op.param().new_shape;
    ir::Shape new_shape =
      shape_inference::inferReshapeShape(input.shape(), shape.data(), shape.size());

    if (new_shape != output.shape())
    {
      // change on output shape
      output.info().shape(new_shape);
    }
  }
  else
  {
    throw std::runtime_error("Reshape: new shape is missing");
  }
}

void StaticShapeInferer::visit(const ir::operation::ResizeBilinear &op)
{
  auto &operands = _lowered_subg->graph().operands();

  const auto input_idx{op.getInputs().at(ir::operation::ResizeBilinear::Input::INPUT)};
  const auto &input = operands.at(input_idx);

  // get mutable output operand
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = operands.at(output_idx);

  int32_t height_out, width_out;
  if (op.getInputs().size() == 2)
  {
    auto &size = operands.at(op.getInputs().at(ir::operation::ResizeBilinear::Input::SIZE));
    if (!size.isConstant())
    {
      output.info().setDynamic();
      return;
    }
    const auto size_v = size.asVector<std::int32_t>();
    height_out = size_v[0];
    width_out = size_v[1];
  }
  else
  {
    height_out = op.param().height_out;
    width_out = op.param().width_out;
  }

  // Shape inferencing logic based on Params
  ir::Shape new_shape =
    shape_inference::inferResizeBilinearShape(input.shape(), height_out, width_out);

  // if size_op is from Const, TFLC put the shape of output into tensor
  if (new_shape != output.shape())
  {
    // change on output shape
    output.info().shape(new_shape);
  }
}

void StaticShapeInferer::visit(const ir::operation::Reverse &op)
{
  handleSimpleUnaryOp(op, op.getInputs().at(ir::operation::Reverse::Input::INPUT));
}

void StaticShapeInferer::visit(const ir::operation::Select &op)
{
  auto &operands = _lowered_subg->graph().operands();

  const auto input_cond_idx{op.getInputs().at(ir::operation::Select::Input::CONDITION)};
  const auto &input_cond = operands.at(input_cond_idx);

  const auto input_true_idx{op.getInputs().at(ir::operation::Select::Input::INPUT_TRUE)};
  const auto &input_true = operands.at(input_true_idx);

  const auto input_false_idx{op.getInputs().at(ir::operation::Select::Input::INPUT_FALSE)};
  const auto &input_false = operands.at(input_false_idx);

  auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = operands.at(output_idx);

  // Select output shpae
  ir::Shape new_shape = shape_inference::inferSelectShape(
    input_cond.info().shape(), input_true.info().shape(), input_false.info().shape());
  output.info().shape(new_shape);
}

void StaticShapeInferer::visit(const ir::operation::Shape &op)
{
  auto &operands = _lowered_subg->graph().operands();

  const auto input_idx{op.getInputs().at(0)};
  const auto &input = operands.at(input_idx);

  // get mutable output operand
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = operands.at(output_idx);

  // re-sizing output shape
  ir::Shape output_shape;
  output_shape.append(input.info().shape().rank());

  output.info().shape(output_shape);
}

void StaticShapeInferer::visit(const ir::operation::Slice &op)
{
  auto &operands = _lowered_subg->graph().operands();

  const auto input_index{op.getInputs().at(ir::operation::Slice::Input::INPUT)};
  const auto &input = operands.at(input_index);
  const auto begins_index{op.getInputs().at(ir::operation::Slice::Input::BEGINS)};
  const auto &begins = operands.at(begins_index);
  const auto sizes_index{op.getInputs().at(ir::operation::Slice::Input::SIZES)};
  const auto &sizes = operands.at(sizes_index);
  const auto output_index = op.getOutputs().at(0);
  ir::Operand &output = operands.at(output_index);

  // Whether input is constant or not does not affect whether output is dynamic or not
  if (!(begins.isConstant() && sizes.isConstant()))
  {
    output.info().setDynamic();
    return;
  }

  auto begins_buf = begins.data()->base();
  auto sizes_buf = sizes.data()->base();

  const auto begins_type = begins.typeInfo().type();
  assert(begins_type == ir::DataType::INT32 || begins_type == ir::DataType::INT64);
  assert(begins_type == sizes.typeInfo().type());

  ir::Shape new_shape =
    (begins_type == ir::DataType::INT32)
      ? shape_inference::inferSliceShape<int32_t>(input.info().shape(),
                                                  reinterpret_cast<const int32_t *>(begins_buf),
                                                  reinterpret_cast<const int32_t *>(sizes_buf))
      : shape_inference::inferSliceShape<int64_t>(input.info().shape(),
                                                  reinterpret_cast<const int64_t *>(begins_buf),
                                                  reinterpret_cast<const int64_t *>(sizes_buf));
  output.info().shape(new_shape);
}

void StaticShapeInferer::visit(const ir::operation::Softmax &op)
{
  handleSimpleUnaryOp(op, op.getInputs().at(ir::operation::Softmax::Input::INPUT));
}

void StaticShapeInferer::visit(const ir::operation::SpaceToBatchND &op)
{
  auto &operands = _lowered_subg->graph().operands();

  const auto output_index = op.getOutputs().at(0);
  const auto input_idx{op.getInputs().at(ir::operation::SpaceToBatchND::Input::INPUT)};
  const auto &block_shape_idx{op.getInputs().at(ir::operation::SpaceToBatchND::Input::BLOCK_SIZE)};
  const auto &padding_idx{op.getInputs().at(ir::operation::SpaceToBatchND::Input::PADDINGS)};

  ir::Operand &output = operands.at(output_index);
  const auto &input = operands.at(input_idx);
  const auto &block_shape = operands.at(block_shape_idx);
  const auto &padding = operands.at(padding_idx);

  // Whether input is constant or not does not affect whether output is dynamic or not
  if (!(block_shape.isConstant() && padding.isConstant()))
  {
    output.info().setDynamic();
    return;
  }

  const auto &input_shape = input.info().shape();
  const auto &block_shape_shape = block_shape.info().shape();
  const auto &padding_shape = padding.info().shape();

  auto block_shape_data = reinterpret_cast<const int32_t *>(block_shape.data()->base());
  auto padding_data = reinterpret_cast<const int32_t *>(padding.data()->base());

  ir::Shape new_shape = shape_inference::inferSpaceToBatchNDShape(
    input_shape, block_shape_shape, padding_shape, block_shape_data, padding_data);

  output.info().shape(new_shape);
}

void StaticShapeInferer::visit(const ir::operation::Split &op)
{
  auto &operands = _lowered_subg->graph().operands();

  const auto input_idx{op.getInputs().at(ir::operation::Split::Input::INPUT)};
  const auto &input = operands.at(input_idx);

  const auto axis_idx{op.getInputs().at(ir::operation::Split::Input::AXIS)};
  const auto &axis = operands.at(axis_idx);

  auto outputs = op.getOutputs();
  if (!axis.isConstant())
  {
    for (auto &&output_idx : outputs)
    {
      ir::Operand &output = operands.at(output_idx);
      output.info().setDynamic();
    }
    return;
  }

  const auto num_splits = op.param().num_splits;

  const auto rank = input.info().shape().rank();
  auto axis_value = axis.asScalar<int32_t>();
  axis_value = axis_value < 0 ? axis_value + rank : axis_value;

  assert(0 <= axis_value && axis_value < rank);

  ir::Shape new_shape =
    shape_inference::inferSplitShape(input.info().shape(), axis_value, num_splits);
  for (auto &&output_idx : outputs)
  {
    ir::Operand &output = operands.at(output_idx);
    output.info().shape(new_shape);
  }
}

void StaticShapeInferer::visit(const ir::operation::SquaredDifference &op)
{
  handleBinaryArithmeticOp(op, op.getInputs().at(ir::operation::SquaredDifference::Input::LHS),
                           op.getInputs().at(ir::operation::SquaredDifference::Input::RHS));
}

void StaticShapeInferer::visit(const ir::operation::Squeeze &op)
{
  auto &operands = _lowered_subg->graph().operands();

  const auto input_idx{op.getInputs().at(ir::operation::Squeeze::Input::INPUT)};
  const auto &input = operands.at(input_idx);

  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = operands.at(output_idx);

  // Squeeze output shpae
  ir::Shape new_shape = shape_inference::inferSqueezeShape(input.info().shape(), op.param());
  output.info().shape(new_shape);
}

void StaticShapeInferer::visit(const ir::operation::StridedSlice &op)
{
  auto &operands = _lowered_subg->graph().operands();

  const auto input_index{op.getInputs().at(ir::operation::StridedSlice::Input::INPUT)};
  const auto &input = operands.at(input_index);
  const auto starts_index{op.getInputs().at(ir::operation::StridedSlice::Input::STARTS)};
  const auto &starts = operands.at(starts_index);
  const auto ends_index{op.getInputs().at(ir::operation::StridedSlice::Input::ENDS)};
  const auto &ends = operands.at(ends_index);
  const auto strides_index{op.getInputs().at(ir::operation::StridedSlice::Input::STRIDES)};
  const auto &strides = operands.at(strides_index);
  const auto output_index = op.getOutputs().at(0);
  ir::Operand &output = operands.at(output_index);

  if (!(starts.isConstant() && ends.isConstant() && strides.isConstant()))
  {
    output.info().setDynamic();
    return;
  }

  const auto begin_mask = op.param().begin_mask;
  const auto end_mask = op.param().end_mask;
  const auto shrink_axis_mask = op.param().shrink_axis_mask;
  const auto rank = input.info().shape().rank();

  auto starts_buf = reinterpret_cast<const uint32_t *>(starts.data()->base());
  auto ends_buf = reinterpret_cast<const uint32_t *>(ends.data()->base());
  auto strides_buf = reinterpret_cast<const uint32_t *>(strides.data()->base());

  auto op_params = shape_inference::buildStridedSliceParams(
    starts_buf, ends_buf, strides_buf, begin_mask, end_mask, shrink_axis_mask, rank);

  ir::Shape new_shape =
    shape_inference::inferStridedSliceShape(input.info().shape(), op_params, rank);
  output.info().shape(new_shape);
}

void StaticShapeInferer::visit(const ir::operation::Tile &op)
{
  auto &operands = _lowered_subg->graph().operands();

  const auto input_idx{op.getInputs().at(ir::operation::Tile::Input::INPUT)};
  const auto &input = operands.at(input_idx);

  const auto multiplier_idx{op.getInputs().at(ir::operation::Tile::Input::MULTIPLES)};
  const auto &multiplier = operands.at(multiplier_idx);

  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = operands.at(output_idx);

  if (!multiplier.isConstant())
  {
    output.info().setDynamic();
    return;
  }

  auto multiplier_buffer = reinterpret_cast<const int32_t *>(multiplier.data()->base());
  assert(multiplier_buffer);

  // re-sizing output shape
  auto new_shape = shape_inference::inferTileShape(input.info().shape(), multiplier_buffer,
                                                   multiplier.shape().num_elements());
  output.info().shape(new_shape);
}

void StaticShapeInferer::visit(const ir::operation::Transpose &op)
{
  auto &operands = _lowered_subg->graph().operands();

  const auto input_idx{op.getInputs().at(ir::operation::Transpose::Input::INPUT)};
  const auto &input = operands.at(input_idx);

  const auto perm_idx{op.getInputs().at(ir::operation::Transpose::Input::PERMUTATION)};
  const auto &perm = operands.at(perm_idx);

  // perm.shape() != ir::Shape{0} means that perm is (n-1...0)
  // TODO This condition changes to perm.num_elements() == 0
  const auto is_regular_transpose = perm.shape() == ir::Shape{0};

  // get mutable output operand
  const auto output_idx = op.getOutputs().at(0);
  auto &output = operands.at(output_idx);
  if (!perm.isConstant() && !is_regular_transpose)
  {
    output.info().setDynamic();
    return;
  }

  ir::Shape new_shape;
  if (is_regular_transpose)
  {
    // Call by (n-1...0)
    new_shape = shape_inference::inferTransposeShape(input.info().shape(), nullptr, 0);
  }
  else
  {
    // Check rank
    if (input.info().shape().rank() != static_cast<int>(perm.info().shape().num_elements()))
    {
      throw std::runtime_error("StaticShapeInferer failed, bad rank size: " +
                               std::to_string(perm.info().shape().num_elements()));
    }

    // set output shape, based on input and params
    const auto perm_buf = reinterpret_cast<const int32_t *>(perm.data()->base());
    new_shape = shape_inference::inferTransposeShape(input.info().shape(), perm_buf,
                                                     perm.shape().num_elements());
  }
  output.info().shape(new_shape);
}

void StaticShapeInferer::visit(const ir::operation::Unpack &op)
{
  auto &operands = _lowered_subg->graph().operands();

  const auto input_idx{op.getInputs().at(0)};
  const auto &input = operands.at(input_idx);
  const auto num = op.param().num;
  const auto rank = input.shape().rank();
  const auto axis = ((op.param().axis < 0) ? rank + op.param().axis : op.param().axis);

  assert(axis < rank);
  if (axis < 0)
  {
    for (int out_tensor_idx = 0; out_tensor_idx < num; out_tensor_idx++)
    {
      const auto output_idx = op.getOutputs().at(out_tensor_idx);
      ir::Operand &output = operands.at(output_idx);
      output.info().setDynamic();
    }
    return;
  }

  ir::Shape new_shape = shape_inference::inferUnpackShape(input.info().shape(), axis, rank);

  // re-sizing output shape
  for (int out_tensor_idx = 0; out_tensor_idx < num; out_tensor_idx++)
  {
    const auto output_idx = op.getOutputs().at(out_tensor_idx);
    ir::Operand &output = operands.at(output_idx);
    output.info().shape(new_shape);
  }
}

void StaticShapeInferer::visit(const ir::operation::While &op)
{
  auto body_input_observer = _subg_input_observers.at(op.param().body_subg_index).get();
  auto cond_input_observer = _subg_input_observers.at(op.param().cond_subg_index).get();
  // re-sizing input shapes of body subgraph
  const auto &inputs = op.getInputs();
  std::vector<ir::OperandInfo> inputs_info;
  const auto &graph = _lowered_subg->graph();
  for (size_t i = 0; i < inputs.size(); ++i)
  {
    const auto &operand_info = graph.operands().at(inputs.at(i)).info();
    inputs_info.emplace_back(operand_info);
  }

  body_input_observer->updateShapes(inputs_info);
  _child_inferers.at(op.param().body_subg_index)->infer();

  // Check whether while operation's shapes are predictable
  // This while op's outputs are also updated in the above function
  // "_child_inferers.at(op.param().body_subg_index)->update()". That means that body's outputs and
  // thils op's outputs must have the same shape. So we can predict whether body subgraphs will
  // change at every step by comparing the shapes of inputs/outputs. If any of shape of body outputs
  // and inputs are different Non-constant operands will be set to dynamic.
  bool check_unpredictable_dynamic = false;
  const auto &updated_outputs = op.getOutputs();
  assert(inputs_info.size() == updated_outputs.size());
  for (size_t i = 0; i < updated_outputs.size(); ++i)
  {
    const auto &input_info = inputs_info.at(i);
    const auto &output_info = graph.operands().at(updated_outputs.at(i)).info();
    if (input_info.isDynamic() != output_info.isDynamic() ||
        input_info.shape() != output_info.shape())
    {
      check_unpredictable_dynamic = true;
      break;
    }
  }

  if (check_unpredictable_dynamic)
  {
    body_input_observer->updateShapes(inputs_info, check_unpredictable_dynamic);
    _child_inferers.at(op.param().body_subg_index)->infer();
  }
  cond_input_observer->updateShapes(inputs_info, check_unpredictable_dynamic);
  _child_inferers.at(op.param().cond_subg_index)->infer();
}

void StaticShapeInferer::visit(const ir::operation::DetectionPostProcess &op)
{
  // TODO: NMS supports very limited input/output size.
  ir::operation::DetectionPostProcess::Param param = op.param();

  auto &operands = _lowered_subg->graph().operands();
  const int num_detected_boxes = param.max_detections * param.max_classes_per_detection;

  const auto output_idx1 = op.getOutputs().at(0);
  auto &output1 = operands.at(output_idx1);
  output1.info().shape({1, num_detected_boxes, 4});

  const auto output_idx2 = op.getOutputs().at(1);
  auto &output2 = operands.at(output_idx2);
  output2.info().shape({1, num_detected_boxes});

  const auto output_idx3 = op.getOutputs().at(2);
  auto &output3 = operands.at(output_idx3);
  output3.info().shape({1, num_detected_boxes});

  const auto output_idx4 = op.getOutputs().at(3);
  auto &output4 = operands.at(output_idx4);
  output4.info().shape({1});
}
void StaticShapeInferer::visit(const ir::operation::Bulk &op)
{
  auto &operands = _lowered_subg->graph().operands();

  // TODO: support multiple inputs/outputs
  const auto input_idx{op.getInputs().at(0)};
  const auto &input = operands.at(input_idx);
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = operands.at(output_idx);

  const auto &cur_input_shape = input.info().shape();
  auto origin_output_shape = op.param().origin_output_shapes[0];

  // TODO: more check for valid batch request
  if ((cur_input_shape.dim(0) < origin_output_shape.dim(0)) ||
      (cur_input_shape.dim(0) % origin_output_shape.dim(0) != 0))
  {
    throw std::runtime_error("StaticShapeInferer " + op.name() + ": Not supported batch size");
  }
  size_t batch_multiplier = cur_input_shape.dim(0) / origin_output_shape.dim(0);

  ir::Shape new_shape;
  new_shape.append(origin_output_shape.dim(0) * batch_multiplier);
  for (int32_t d = 1; d < origin_output_shape.rank(); ++d)
    new_shape.append(origin_output_shape.dim(d));

  output.info().shape(new_shape);
}

} // namespace compiler

} // namespace onert
