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

#include <sstream>

namespace onert
{
namespace compiler
{

bool StaticShapeInferer::infer(const ir::OpSequence &op_seq)
{
  bool has_dynamic_tensor = false;

  for (const auto &operation_idx : op_seq.operations())
  {
    auto &op = _operations.at(operation_idx);
    auto opcode = op.opcode();

    _return_has_dynamic_tensor = false; // this is used as a return value inside operation's visit()

    // IF: need shape inference for then, else
    // While: need shape inference for condition, body
    if (opcode == ir::OpCode::If || opcode == ir::OpCode::While)
    {
      op.accept(*this);
    }
    else
    {
      _return_has_dynamic_tensor = checkDynamicInput(op);

      if (_return_has_dynamic_tensor)
      {
        setDynamicOutput(op);
      }
      else
      {
        op.accept(*this);
      }
    }

    has_dynamic_tensor = has_dynamic_tensor || _return_has_dynamic_tensor;
  }

  return has_dynamic_tensor;
}

bool StaticShapeInferer::checkDynamicInput(const ir::Operation &op)
{
  for (auto input_idx : op.getInputs() | ir::Remove::UNDEFINED | ir::Remove::DUPLICATED)
  {
    if (_operands.at(input_idx).info().isDynamic())
    {
      return true;
    }
  }

  return false;
}

void StaticShapeInferer::setDynamicOutput(const ir::Operation &op)
{
  for (auto output_idx : op.getOutputs() | ir::Remove::UNDEFINED)
  {
    _operands.at(output_idx).info().setDynamic();
  }
}

void StaticShapeInferer::handleBinaryArithmeticOp(const ir::Operation &op,
                                                  const ir::OperandIndex lhs_idx,
                                                  const ir::OperandIndex rhs_idx)
{
  const auto &lhs = _operands.at(lhs_idx);
  const auto &rhs = _operands.at(rhs_idx);

  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

  // re-sizing output shape
  ir::Shape new_shape = shape_inference::inferEltwiseShape(lhs.info().shape(), rhs.info().shape());
  output.info().shape(new_shape);
}

void StaticShapeInferer::handleSimpleUnaryOp(const ir::Operation &op,
                                             const ir::OperandIndex input_idx)
{
  const auto &input = _operands.at(input_idx);

  // get mutable output operand
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

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

  for (const auto &pair : _lowered_subgs)
  {
    const auto index = pair.first;
    const auto &lowered_subg = pair.second;
    VERBOSE(StaticShapeInferer) << "SubGraph #" << index.value() << std::endl;
    lowered_subg->graph().operands().iterate(
        [&](const ir::OperandIndex &ind, const ir::Operand &operand) {
          VERBOSE(StaticShapeInferer) << "Operand #" << ind.value() << ", "
                                      << (operand.info().isDynamic() ? "Dynamic" : "Static") << ", "
                                      << get_shape_str(operand.info().shape()) << std::endl;
        });
  }
}

void StaticShapeInferer::visit(const ir::operation::ArgMax &op)
{
  const auto input_idx{op.getInputs().at(ir::operation::ArgMax::Input::INPUT)};
  const auto &input = _operands.at(input_idx);

  const auto axis_idx{op.getInputs().at(ir::operation::ArgMax::Input::AXIS)};
  const auto &axis = _operands.at(axis_idx);

  // get mutable output operand
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

  if (!axis.isConstant())
  {
    output.info().setDynamic();
    _return_has_dynamic_tensor = true;
    return;
  }

  const auto rank = input.info().shape().rank();
  auto axis_value = axis.asScalar<int32_t>();
  axis_value = axis_value < 0 ? axis_value + rank : axis_value;

  // re-sizing output shape
  ir::Shape new_shape = shape_inference::inferArgMaxShape(input.info().shape(), axis_value, rank);
  output.info().shape(new_shape);
}

void StaticShapeInferer::visit(const ir::operation::BatchMatMul &op)
{
  const auto lhs_index = op.getInputs().at(ir::operation::BatchMatMul::Input::LHS);
  const auto rhs_index = op.getInputs().at(ir::operation::BatchMatMul::Input::RHS);
  const auto output_index = op.getOutputs().at(0);
  const auto &lhs = _operands.at(lhs_index);
  const auto &rhs = _operands.at(rhs_index);
  auto &output = _operands.at(output_index);
  auto new_shape = shape_inference::inferBatchMatMulShape(lhs.shape(), rhs.shape(), op.param());
  output.info().shape(new_shape);
}

void StaticShapeInferer::visit(const ir::operation::BCQFullyConnected &op)
{
  const auto input_idx{op.getInputs().at(ir::operation::BCQFullyConnected::Input::INPUT)};
  const auto &input = _operands.at(input_idx);

  const auto cluster_idx{
      op.getInputs().at(ir::operation::BCQFullyConnected::Input::WEIGHTS_CLUSTERS)};
  const auto &cluster = _operands.at(cluster_idx);

  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

  auto cluster_buf = reinterpret_cast<const int32_t *>(cluster.data()->base());
  assert(cluster_buf);

  // re-sizing output shape
  ir::Shape new_shape = shape_inference::inferBCQFullyConnectedShape(
      input.info().shape(), cluster.info().shape(), cluster_buf);
  output.info().shape(new_shape);
}

void StaticShapeInferer::visit(const ir::operation::BCQGather &op)
{
  const auto indices_idx{op.getInputs().at(ir::operation::BCQGather::Input::INDICES)};
  const auto &indices = _operands.at(indices_idx);

  const auto input_binary_idx{op.getInputs().at(ir::operation::BCQGather::Input::INPUT_BINARY)};
  const auto &input_binary = _operands.at(input_binary_idx);

  const auto cluster_idx{op.getInputs().at(ir::operation::BCQGather::Input::INPUT_CLUSTERS)};
  const auto &cluster = _operands.at(cluster_idx);

  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

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
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

  const auto shape_idx{op.getInputs().at(ir::operation::BroadcastTo::Input::SHAPE)};
  const auto &shape = _operands.at(shape_idx);

  if (!shape.isConstant())
  {
    output.info().setDynamic();
    _return_has_dynamic_tensor = true;
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
  const auto input_count = op.getInputs().size();

  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

  shape_inference::Shapes input_shapes;
  for (uint32_t i = 0; i < input_count; i++)
  {
    const auto input_idx{op.getInputs().at(i)};
    const auto &input = _operands.at(input_idx);
    input_shapes.emplace_back(input.shape());
  }

  ir::Shape out_shape = shape_inference::inferConcatShape(input_shapes, op.param());

  // re-sizing output shape
  output.info().shape(out_shape);
}

void StaticShapeInferer::visit(const ir::operation::Conv2D &op)
{
  const auto input_idx{op.getInputs().at(ir::operation::Conv2D::Input::INPUT)};
  const auto &input = _operands.at(input_idx);
  const auto ker_idx{op.getInputs().at(ir::operation::Conv2D::Input::KERNEL)};
  const auto &ker = _operands.at(ker_idx);
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

  // re-sizing output shape
  ir::Shape new_shape =
      shape_inference::inferConv2DShape(input.info().shape(), ker.info().shape(), op.param());
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
  const auto input_idx{op.getInputs().at(ir::operation::ExpandDims::Input::INPUT)};
  const auto &input = _operands.at(input_idx);
  const auto axis_idx{op.getInputs().at(ir::operation::ExpandDims::Input::AXIS)};
  const auto &axis = _operands.at(axis_idx);
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

  if (!axis.isConstant())
  {
    output.info().setDynamic();
    _return_has_dynamic_tensor = true;
    return;
  }

  // even when axis is constant, output shape should be recalculated since user might call
  // nnfw_set_input_tensorinfo(input, some_new_shape)
  auto axis_buf = reinterpret_cast<const int32_t *>(axis.data()->base());
  assert(axis_buf);

  // re-sizing output shape
  ir::Shape new_shape = shape_inference::inferExpandDimsShape(input.info().shape(), axis_buf[0]);
  output.info().shape(new_shape);
}

void StaticShapeInferer::visit(const ir::operation::Fill &op)
{
  const auto input_idx{op.getInputs().at(ir::operation::Fill::Input::INPUT)};
  const auto &input = _operands.at(input_idx);
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

  if (!input.isConstant())
  {
    output.info().setDynamic();
    _return_has_dynamic_tensor = true;
    return;
  }

  assert(input.typeInfo().type() == ir::DataType::INT32);

  auto input_buf = reinterpret_cast<const int32_t *>(input.data()->base());
  assert(input_buf);

  // re-sizing output shape
  ir::Shape new_shape = shape_inference::inferFillShape(input.info().shape(), input_buf);
  output.info().shape(new_shape);
}

void StaticShapeInferer::visit(const ir::operation::FullyConnected &op)
{
  const auto input_idx{op.getInputs().at(ir::operation::FullyConnected::Input::INPUT)};
  const auto &input = _operands.at(input_idx);

  const auto ker_idx{op.getInputs().at(ir::operation::FullyConnected::Input::WEIGHT)};
  const auto &ker = _operands.at(ker_idx);

  // get mutable output operand
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);
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
  const auto input_idx{op.getInputs().at(ir::operation::Gather::Input::INPUT)};
  const auto &input = _operands.at(input_idx);

  // get mutable output operand
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

  const auto indices_idx{op.getInputs().at(ir::operation::Gather::Input::INDICES)};
  const auto &indices = _operands.at(indices_idx);
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
  auto &then_graph = _lowered_subgs.at(op.param().then_subg_index)->graph();
  auto &else_graph = _lowered_subgs.at(op.param().else_subg_index)->graph();
  const std::vector<ir::OperandIndex> inputs{op.getInputs().begin() + 1, op.getInputs().end()};
  const auto &outputs = op.getOutputs();

  // re-sizing input shapes of then subgraph
  const auto &then_inputs = then_graph.getInputs();
  assert(inputs.size() == then_inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i)
  {
    auto &then_input = then_graph.operands().at(then_inputs.at(i));
    if (_operands.at(inputs.at(i)).info().isDynamic())
    {
      then_input.info().setDynamic();
    }
    else
    {
      auto new_shape = _operands.at(inputs.at(i)).info().shape();
      then_input.info().shape(new_shape);
    }
  }

  // re-sizing input shapes of else subgraph
  const auto &else_inputs = else_graph.getInputs();
  assert(inputs.size() == else_inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i)
  {
    auto &else_input = else_graph.operands().at(else_inputs.at(i));
    if (_operands.at(inputs.at(i)).info().isDynamic())
    {
      else_input.info().setDynamic();
    }
    else
    {
      const auto &new_shape = _operands.at(inputs.at(i)).info().shape();
      else_input.info().shape(new_shape);
    }
  }

  // re-sizing operands of then subgraph
  StaticShapeInferer then_inferer(op.param().then_subg_index, _lowered_subgs);
  _lowered_subgs.at(op.param().then_subg_index)
      ->iterateTopolOpSeqs([&](const ir::OpSequenceIndex &, ir::OpSequence &op_seq) {
        bool has_dynamic_tensor = then_inferer.infer(op_seq);
        op_seq.has_dynamic_tensor(has_dynamic_tensor);
      });

  // re-sizing operands of else subgraph
  StaticShapeInferer else_inferer(op.param().else_subg_index, _lowered_subgs);
  _lowered_subgs.at(op.param().else_subg_index)
      ->iterateTopolOpSeqs([&](const ir::OpSequenceIndex &, ir::OpSequence &op_seq) {
        bool has_dynamic_tensor = else_inferer.infer(op_seq);
        op_seq.has_dynamic_tensor(has_dynamic_tensor);
      });

  // re-sizing output shapes
  const auto &then_outputs = _lowered_subgs.at(op.param().then_subg_index)->graph().getOutputs();
  const auto &else_outputs = _lowered_subgs.at(op.param().else_subg_index)->graph().getOutputs();
  assert(outputs.size() == then_outputs.size());
  assert(outputs.size() == else_outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i)
  {
    const auto &then_output = then_graph.operands().at(then_outputs.at(i));
    const auto &else_output = else_graph.operands().at(else_outputs.at(i));
    auto &output = _operands.at(outputs.at(i));
    if (!then_output.info().isDynamic() && !else_output.info().isDynamic() &&
        then_output.shape() == else_output.shape())
    {
      output.info().shape(then_output.shape());
    }
    else
    {
      output.info().setDynamic();
      _return_has_dynamic_tensor = true;
    }
  }
}

void StaticShapeInferer::visit(const ir::operation::L2Normalization &op)
{
  handleSimpleUnaryOp(op, op.getInputs().at(ir::operation::L2Normalization::Input::INPUT));
}

void StaticShapeInferer::visit(const ir::operation::LSTM &op)
{
  const auto output_index{op.getOutputs().at(ir::operation::LSTM::Output::OUTPUT)};
  auto &output = _operands.at(output_index);

  const auto output_state_out_index{
      op.getOutputs().at(ir::operation::LSTM::Output::OUTPUT_STATE_OUT)};

  const auto cell_state_out_index{op.getOutputs().at(ir::operation::LSTM::Output::CELL_STATE_OUT)};

  const auto scratch_buffer_index{op.getOutputs().at(ir::operation::LSTM::Output::SCRATCH_BUFFER)};

  if (output.info().isDynamic() || (_operands.exist(output_state_out_index) &&
                                    _operands.at(output_state_out_index).info().isDynamic()) ||
      (_operands.exist(cell_state_out_index) &&
       _operands.at(cell_state_out_index).info().isDynamic()) ||
      (_operands.exist(scratch_buffer_index) &&
       _operands.at(scratch_buffer_index).info().isDynamic()))
    return;

  const auto input_index{op.getInputs().at(ir::operation::LSTM::Input::INPUT)};
  const auto &input = _operands.at(input_index);

  const auto input_to_output_weights_index{
      op.getInputs().at(ir::operation::LSTM::Input::INPUT_TO_OUTPUT_WEIGHTS)};
  const auto &input_to_output_weights = _operands.at(input_to_output_weights_index);

  const auto recurrent_to_output_weights_index{
      op.getInputs().at(ir::operation::LSTM::Input::RECURRENT_TO_OUTPUT_WEIGHTS)};
  const auto &recurrent_to_output_weights = _operands.at(recurrent_to_output_weights_index);

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

  if (_operands.exist(output_state_out_index))
  {
    auto &output_state_out = _operands.at(output_state_out_index);
    output_state_out.info().shape(ir::Shape{n_batch, n_output});
  }

  if (_operands.exist(cell_state_out_index))
  {
    auto &cell_state_out = _operands.at(cell_state_out_index);
    cell_state_out.info().shape(ir::Shape{n_batch, n_cell});
  }

  if (_operands.exist(scratch_buffer_index))
  {
    auto &scratch_buffer = _operands.at(scratch_buffer_index);

    const auto input_to_input_weights_index{
        op.getInputs().at(ir::operation::LSTM::Input::INPUT_TO_INPUT_WEIGHTS)};
    const auto recurrent_to_input_weights_index{
        op.getInputs().at(ir::operation::LSTM::Input::RECURRENT_TO_INPUT_WEIGHTS)};

    bool has_input_to_input_weights =
        _operands.at(input_to_input_weights_index).shape().dim(0) != 0 &&
        _operands.at(input_to_input_weights_index).shape().dim(1) != 0;
    bool has_recurrent_to_input_weights =
        _operands.at(recurrent_to_input_weights_index).shape().dim(0) != 0 &&
        _operands.at(recurrent_to_input_weights_index).shape().dim(1) != 0;

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
  const auto indice_idx{op.getInputs().at(ir::operation::OneHot::Input::INDICES)};
  const auto &indice = _operands.at(indice_idx);
  const auto depth_idx{op.getInputs().at(ir::operation::OneHot::Input::DEPTH)};
  const auto &depth = _operands.at(depth_idx);

  const auto axis = op.param().axis;

  auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

  if (!depth.isConstant())
  {
    output.info().setDynamic();
    _return_has_dynamic_tensor = true;
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
  const auto input_idx{op.getInputs().at(0)};
  const auto &input = _operands.at(input_idx);

  // get mutable output operand
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

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
  const auto input_idx{op.getInputs().at(ir::operation::Pad::Input::INPUT)};
  const auto &input = _operands.at(input_idx);

  const auto pad_idx{op.getInputs().at(ir::operation::Pad::Input::PAD)};
  const auto &pad = _operands.at(pad_idx);

  // get mutable output operand
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

  // if pad is not constant, output also becomes dynamic
  if (!pad.isConstant())
  {
    output.info().setDynamic();
    _return_has_dynamic_tensor = true;
    return;
  }

  // re-sizing output shape
  const auto new_shape = shape_inference::inferPadShape(
      input.shape(), reinterpret_cast<const int32_t *>(pad.data()->base()),
      pad.shape().num_elements());
  output.info().shape(new_shape);
}

void StaticShapeInferer::visit(const ir::operation::Permute &op)
{
  const auto input_idx{op.getInputs().at(0)};
  const auto &input = _operands.at(input_idx);
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

  // re-sizing output shape
  // Permute is a special operation that layouts of input/output may be different on backend
  // However, it is not applied here, so input/output have the same layout of frontend. Because
  // "ExecutorFactory" would convert shape of input/output accoding to the layouts when registering
  // operand info to "TensorBuilder" after calling "StaticShapeInferer"
  const auto new_shape = input.info().shape();
  output.info().shape(new_shape);
}

void StaticShapeInferer::visit(const ir::operation::Pow &op)
{
  handleBinaryArithmeticOp(op, op.getInputs().at(ir::operation::Pow::Input::LHS),
                           op.getInputs().at(ir::operation::Pow::Input::RHS));
}

void StaticShapeInferer::visit(const ir::operation::Range &op)
{
  const auto start_idx{op.getInputs().at(ir::operation::Range::Input::START)};
  const auto limit_idx{op.getInputs().at(ir::operation::Range::Input::LIMIT)};
  const auto delta_idx{op.getInputs().at(ir::operation::Range::Input::DELTA)};
  const auto &start_op = _operands.at(start_idx);
  const auto &limit_op = _operands.at(limit_idx);
  const auto &delta_op = _operands.at(delta_idx);

  // get mutable output operand
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

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
    _return_has_dynamic_tensor = true;
  }
}

void StaticShapeInferer::visit(const ir::operation::Reduce &op)
{
  const auto input_idx{op.getInputs().at(ir::operation::Reduce::Input::INPUT)};
  const auto &input = _operands.at(input_idx);

  const auto axes_idx{op.getInputs().at(ir::operation::Reduce::Input::AXES)};
  const auto &axes = _operands.at(axes_idx);

  // get mutable output operand
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

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
  const auto input_idx{op.getInputs().at(ir::operation::Reshape::Input::INPUT)};
  const auto &input = _operands.at(input_idx);

  // get mutable output operand
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

  // New shape is given by second input tensor
  if (op.getInputs().size() == 2)
  {
    // Let's check the second input
    const auto shape_idx{op.getInputs().at(ir::operation::Reshape::Input::SHAPE)};
    const auto &shape = _operands.at(shape_idx);

    if (shape.isConstant())
    {
      const auto *shape_buf = reinterpret_cast<const int32_t *>(shape.data()->base());
      assert(shape_buf);

      ir::Shape new_shape = shape_inference::inferReshapeShape(
          shape_buf, shape.shape().num_elements(), input.shape().num_elements());

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
      _return_has_dynamic_tensor = true;
    }
  }
  // New shape is given by option
  else if (op.param().new_shape.size() != 0)
  {
    // Let's check the new_shape option
    auto shape = op.param().new_shape;
    ir::Shape new_shape = shape_inference::inferReshapeShape(shape.data(), shape.size(),
                                                             input.shape().num_elements());

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
  const auto input_idx{op.getInputs().at(ir::operation::ResizeBilinear::Input::INPUT)};
  const auto &input = _operands.at(input_idx);

  // get mutable output operand
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

  int32_t height_out, width_out;
  if (op.getInputs().size() == 2)
  {
    auto &size = _operands.at(op.getInputs().at(ir::operation::ResizeBilinear::Input::SIZE));
    if (!size.isConstant())
    {
      output.info().setDynamic();
      _return_has_dynamic_tensor = true;
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
  const auto input_cond_idx{op.getInputs().at(ir::operation::Select::Input::CONDITION)};
  const auto &input_cond = _operands.at(input_cond_idx);

  const auto input_true_idx{op.getInputs().at(ir::operation::Select::Input::INPUT_TRUE)};
  const auto &input_true = _operands.at(input_true_idx);

  const auto input_false_idx{op.getInputs().at(ir::operation::Select::Input::INPUT_FALSE)};
  const auto &input_false = _operands.at(input_false_idx);

  auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

  // Select output shpae
  ir::Shape new_shape = shape_inference::inferSelectShape(
      input_cond.info().shape(), input_true.info().shape(), input_false.info().shape());
  output.info().shape(new_shape);
}

void StaticShapeInferer::visit(const ir::operation::Shape &op)
{
  const auto input_idx{op.getInputs().at(0)};
  const auto &input = _operands.at(input_idx);

  // get mutable output operand
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

  // re-sizing output shape
  ir::Shape output_shape;
  output_shape.append(input.info().shape().rank());

  output.info().shape(output_shape);
}

void StaticShapeInferer::visit(const ir::operation::Slice &op)
{
  const auto input_index{op.getInputs().at(ir::operation::Slice::Input::INPUT)};
  const auto &input = _operands.at(input_index);
  const auto begins_index{op.getInputs().at(ir::operation::Slice::Input::BEGINS)};
  const auto &begins = _operands.at(begins_index);
  const auto sizes_index{op.getInputs().at(ir::operation::Slice::Input::SIZES)};
  const auto &sizes = _operands.at(sizes_index);
  const auto output_index = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_index);

  // Whether input is constant or not does not affect whether output is dynamic or not
  if (!(begins.isConstant() && sizes.isConstant()))
  {
    output.info().setDynamic();
    _return_has_dynamic_tensor = true;
    return;
  }

  auto begins_buf = reinterpret_cast<const int32_t *>(begins.data()->base());
  auto sizes_buf = reinterpret_cast<const int32_t *>(sizes.data()->base());

  ir::Shape new_shape =
      shape_inference::inferSliceShape(input.info().shape(), begins_buf, sizes_buf);
  output.info().shape(new_shape);
}

void StaticShapeInferer::visit(const ir::operation::Softmax &op)
{
  handleSimpleUnaryOp(op, op.getInputs().at(ir::operation::Softmax::Input::INPUT));
}

void StaticShapeInferer::visit(const ir::operation::SpaceToBatchND &op)
{
  const auto output_index = op.getOutputs().at(0);
  const auto input_idx{op.getInputs().at(ir::operation::SpaceToBatchND::Input::INPUT)};
  const auto block_shape_idx{op.getInputs().at(ir::operation::SpaceToBatchND::Input::BLOCK_SIZE)};
  const auto padding_idx{op.getInputs().at(ir::operation::SpaceToBatchND::Input::PADDINGS)};

  ir::Operand &output = _operands.at(output_index);
  const auto &input = _operands.at(input_idx);
  const auto &block_shape = _operands.at(block_shape_idx);
  const auto &padding = _operands.at(padding_idx);

  // Whether input is constant or not does not affect whether output is dynamic or not
  if (!(block_shape.isConstant() && padding.isConstant()))
  {
    output.info().setDynamic();
    _return_has_dynamic_tensor = true;
    return;
  }

  auto input_shape = input.info().shape();
  auto block_shape_shape = block_shape.info().shape();
  auto padding_shape = padding.info().shape();

  auto block_shape_data = reinterpret_cast<const int32_t *>(block_shape.data()->base());
  auto padding_data = reinterpret_cast<const int32_t *>(padding.data()->base());

  ir::Shape new_shape = shape_inference::inferSpaceToBatchNDShape(
      input_shape, block_shape_shape, padding_shape, block_shape_data, padding_data);

  output.info().shape(new_shape);
}

void StaticShapeInferer::visit(const ir::operation::Split &op)
{
  const auto input_idx{op.getInputs().at(ir::operation::Split::Input::INPUT)};
  const auto &input = _operands.at(input_idx);

  const auto axis_idx{op.getInputs().at(ir::operation::Split::Input::AXIS)};
  const auto &axis = _operands.at(axis_idx);

  auto outputs = op.getOutputs();
  if (!axis.isConstant())
  {
    for (auto output_idx : outputs)
    {
      ir::Operand &output = _operands.at(output_idx);
      output.info().setDynamic();
    }
    _return_has_dynamic_tensor = true;
    return;
  }

  const auto num_splits = op.param().num_splits;

  const auto rank = input.info().shape().rank();
  auto axis_value = axis.asScalar<int32_t>();
  axis_value = axis_value < 0 ? axis_value + rank : axis_value;

  assert(0 <= axis_value && axis_value < rank);

  ir::Shape new_shape =
      shape_inference::inferSplitShape(input.info().shape(), axis_value, num_splits);
  for (auto output_idx : outputs)
  {
    ir::Operand &output = _operands.at(output_idx);
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
  const auto input_idx{op.getInputs().at(ir::operation::Squeeze::Input::INPUT)};
  const auto &input = _operands.at(input_idx);

  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

  // Squeeze output shpae
  ir::Shape new_shape = shape_inference::inferSqueezeShape(input.info().shape(), op.param());
  output.info().shape(new_shape);
}

void StaticShapeInferer::visit(const ir::operation::StridedSlice &op)
{
  const auto input_index{op.getInputs().at(ir::operation::StridedSlice::Input::INPUT)};
  const auto &input = _operands.at(input_index);
  const auto starts_index{op.getInputs().at(ir::operation::StridedSlice::Input::STARTS)};
  const auto &starts = _operands.at(starts_index);
  const auto ends_index{op.getInputs().at(ir::operation::StridedSlice::Input::ENDS)};
  const auto &ends = _operands.at(ends_index);
  const auto strides_index{op.getInputs().at(ir::operation::StridedSlice::Input::STRIDES)};
  const auto &strides = _operands.at(strides_index);
  const auto output_index = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_index);

  if (!(starts.isConstant() && ends.isConstant() && strides.isConstant()))
  {
    output.info().setDynamic();
    _return_has_dynamic_tensor = true;
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
  const auto input_idx{op.getInputs().at(ir::operation::Tile::Input::INPUT)};
  const auto &input = _operands.at(input_idx);

  const auto multiplier_idx{op.getInputs().at(ir::operation::Tile::Input::MULTIPLES)};
  const auto &multiplier = _operands.at(multiplier_idx);

  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

  if (!multiplier.isConstant())
  {
    output.info().setDynamic();
    _return_has_dynamic_tensor = true;
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
  const auto input_idx{op.getInputs().at(ir::operation::Transpose::Input::INPUT)};
  const auto &input = _operands.at(input_idx);

  const auto perm_idx{op.getInputs().at(ir::operation::Transpose::Input::PERMUTATION)};
  const auto &perm = _operands.at(perm_idx);

  // perm.shape() != ir::Shape{0} means that perm is (n-1...0)
  // TODO This condition changes to perm.num_elements() == 0
  const auto is_regular_transpose = perm.shape() == ir::Shape{0};

  // get mutable output operand
  const auto output_idx = op.getOutputs().at(0);
  auto &output = _operands.at(output_idx);
  if (!perm.isConstant() && !is_regular_transpose)
  {
    output.info().setDynamic();
    _return_has_dynamic_tensor = true;
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
  const auto input_idx{op.getInputs().at(0)};
  const auto &input = _operands.at(input_idx);
  const auto num = op.param().num;
  const auto rank = input.shape().rank();
  const auto axis = ((op.param().axis < 0) ? rank + op.param().axis : op.param().axis);

  assert(axis < rank);
  if (axis < 0)
  {
    for (int out_tensor_idx = 0; out_tensor_idx < num; out_tensor_idx++)
    {
      const auto output_idx = op.getOutputs().at(out_tensor_idx);
      ir::Operand &output = _operands.at(output_idx);
      output.info().setDynamic();
    }
    _return_has_dynamic_tensor = true;
    return;
  }

  ir::Shape new_shape = shape_inference::inferUnpackShape(input.info().shape(), axis, rank);

  // re-sizing output shape
  for (int out_tensor_idx = 0; out_tensor_idx < num; out_tensor_idx++)
  {
    const auto output_idx = op.getOutputs().at(out_tensor_idx);
    ir::Operand &output = _operands.at(output_idx);
    output.info().shape(new_shape);
  }
}

void StaticShapeInferer::visit(const ir::operation::While &op)
{
  auto &cond_graph = _lowered_subgs.at(op.param().cond_subg_index)->graph();
  auto &body_graph = _lowered_subgs.at(op.param().body_subg_index)->graph();
  const auto inputs = op.getInputs();
  const auto &outputs = op.getOutputs();

  // re-sizing input shapes of then subgraph
  const auto &cond_inputs = cond_graph.getInputs();
  assert(inputs.size() == cond_inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i)
  {
    const auto &input = _operands.at(inputs.at(i));
    auto &cond_input = cond_graph.operands().at(cond_inputs.at(i));
    if (input.info().isDynamic())
    {
      cond_input.info().setDynamic();
    }
    else
    {
      auto new_shape = input.info().shape();
      cond_input.info().shape(new_shape);
    }
  }

  body_graph.operands().iterate([](const onert::ir::OperandIndex &idx, onert::ir::Operand &) {
    VERBOSE(BODY_GRAPH) << idx << std::endl;
  });
  // re-sizing input shapes of body subgraph
  const auto &body_inputs = body_graph.getInputs();
  assert(cond_inputs.size() == body_inputs.size());
  for (size_t i = 0; i < cond_inputs.size(); ++i)
  {
    const auto &cond_input = cond_graph.operands().at(cond_inputs.at(i));
    auto &body_input = body_graph.operands().at(body_inputs.at(i));
    if (cond_input.info().isDynamic())
    {
      body_input.info().setDynamic();
    }
    else
    {
      const auto &new_shape = cond_input.info().shape();
      body_input.info().shape(new_shape);
    }
  }

  // re-sizing operands of body subgraph
  StaticShapeInferer body_inferer(op.param().body_subg_index, _lowered_subgs);
  _lowered_subgs.at(op.param().body_subg_index)
      ->iterateTopolOpSeqs([&](const ir::OpSequenceIndex &, ir::OpSequence &op_seq) {
        bool has_dynamic_tensor = body_inferer.infer(op_seq);
        op_seq.has_dynamic_tensor(has_dynamic_tensor);
      });

  // Check whether while operation's shapes are predictable
  // If any of shape of body outputs and cond inputs are different, non-constant operands would be
  // set to dynamic
  bool check_unpredictable_dynamic = false;
  const auto &body_outputs = body_graph.getOutputs();
  assert(body_outputs.size() == cond_inputs.size());
  for (size_t i = 0; i < body_outputs.size(); ++i)
  {
    const auto &body_output = body_graph.operands().at(body_outputs.at(i));
    auto &cond_input = cond_graph.operands().at(cond_inputs.at(i));
    if ((cond_input.info().isDynamic() != body_output.info().isDynamic()) ||
        (cond_input.shape() != body_output.shape()))
    {
      check_unpredictable_dynamic = true;
      break;
    }
  }

  if (check_unpredictable_dynamic)
  {
    // Set inputs of body subgraph
    for (const auto &input_index : body_inputs)
    {
      auto &input = body_graph.operands().at(input_index);
      if (!input.isConstant())
      {
        input.info().setDynamic();
      }
    }

    // Set inputs of cond subgraph
    for (const auto &input_index : cond_inputs)
    {
      auto &input = cond_graph.operands().at(input_index);
      if (!input.isConstant())
      {
        input.info().setDynamic();
      }
    }

    // Set non-constant operands of body subgraph to dynamic
    StaticShapeInferer body_inferer(op.param().body_subg_index, _lowered_subgs);
    _lowered_subgs.at(op.param().body_subg_index)
        ->iterateTopolOpSeqs([&](const ir::OpSequenceIndex &, ir::OpSequence &op_seq) {
          bool has_dynamic_tensor = body_inferer.infer(op_seq);
          op_seq.has_dynamic_tensor(has_dynamic_tensor);
        });
  }

  // re-sizing operands of cond subgraph
  // If check_unpredictable_dynamic is true, non-constant operands of cond subgraph would be set to
  // dynamic
  StaticShapeInferer cond_inferer(op.param().cond_subg_index, _lowered_subgs);
  _lowered_subgs.at(op.param().cond_subg_index)
      ->iterateTopolOpSeqs([&](const ir::OpSequenceIndex &, ir::OpSequence &op_seq) {
        bool has_dynamic_tensor = cond_inferer.infer(op_seq);
        op_seq.has_dynamic_tensor(has_dynamic_tensor);
      });

  // re-sizing outputs of while operation
  // If check_unpredictable_dynamic is true, outputs of while operation would be set to dynamic
  assert(cond_inputs.size() == outputs.size());
  for (size_t i = 0; i < cond_inputs.size(); ++i)
  {
    const auto &cond_input = cond_graph.operands().at(cond_inputs.at(i));
    auto &output = _operands.at(outputs.at(i));
    if (cond_input.info().isDynamic())
    {
      output.info().setDynamic();
      _return_has_dynamic_tensor = true;
    }
    else
    {
      const auto new_shape = cond_input.info().shape();
      output.info().shape(new_shape);
    }
  }
}

} // namespace compiler

} // namespace onert
