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

#include "exec/DynamicShapeInference.h"
#include "util/ShapeInference.h"

namespace onert
{
namespace exec
{

void DynamicShapeInferer::handleBinaryArithmeticOp(const ir::Operation &op,
                                                   const ir::OperandIndex lhs_idx,
                                                   const ir::OperandIndex rhs_idx)
{
  auto lhs = _tensor_registry->getITensor(lhs_idx);
  auto lhs_shape = lhs->getShape();

  auto rhs = _tensor_registry->getITensor(rhs_idx);
  auto rhs_shape = rhs->getShape();

  /*
    Here, the state after compilation (satic shape inference) could be one of the following:

              lhs       rhs              output     execution-time shape inf required
      ------------------------------------------    ---------------------------------
      case 1) static    static           static      X
      case 2) one or both are dynamic    dynamic     O

    Then nnfw_apply_tensorinf() could change one or both inputs dynamic.
    So, in this method, we have one more state and we have to re-calculate shape for this shape.

      case 3) one or both are dynamic    static      O

    So, only when all inputs are static, we can skip dynamic shape inference.
  */
  if ((!lhs->is_dynamic()) && (!rhs->is_dynamic()))
    return;

  auto output_idx = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_idx);

  ir::Shape new_shape = shape_inference::inferEltwiseShape(lhs_shape, rhs_shape);

  _dynamic_tensor_manager->applyShape(output_idx, new_shape);
  assert(output->buffer() != nullptr);
}

void DynamicShapeInferer::handleSimpleUnaryOp(const ir::Operation &op,
                                              const ir::OperandIndex input_ind)
{
  // check if input is not dynamic
  auto input = _tensor_registry->getITensor(input_ind);
  auto output_shape = input->getShape();

  /*
    Here, the state after compilation (satic shape inference) could be one of the following:

              input      output    execution-time shape inf required
      -------------------------    ---------------------------------
      case 1) static     static      X
      case 2) dynamic    dynamic     O

    Then nnfw_apply_tensorinf() could change input dynamic.
    So, in this method, we have one more state and we have to re-calculate shape for this shape.

      case 3) dynamic    static      O

    So, only when input is static, we can skip dynamic shape inference.
  */
  if (!input->is_dynamic())
    return;

  auto output_ind = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_ind);

  _dynamic_tensor_manager->applyShape(output_ind, output_shape);
  assert(output->buffer() != nullptr);
}

void DynamicShapeInferer::visit(const ir::operation::Abs &op)
{
  handleSimpleUnaryOp(op, op.getInputs().at(ir::operation::Abs::INPUT));
}

void DynamicShapeInferer::visit(const ir::operation::Add &op)
{
  handleBinaryArithmeticOp(op, op.getInputs().at(ir::operation::Add::Input::LHS),
                           op.getInputs().at(ir::operation::Add::Input::RHS));
}

void DynamicShapeInferer::visit(const ir::operation::ArgMax &op)
{
  const auto input_idx{op.getInputs().at(ir::operation::ArgMax::Input::INPUT)};
  const auto &input = _tensor_registry->getITensor(input_idx);
  auto input_shape = input->getShape();

  if (!input->is_dynamic())
    return;

  const auto rank = input_shape.rank();
  const auto axis = ((op.param().axis < 0) ? rank + op.param().axis : op.param().axis);

  assert(0 <= axis && axis < rank);

  auto output_ind = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_ind);

  ir::Shape new_shape = shape_inference::inferArgMaxShape(input_shape, axis, rank);

  _dynamic_tensor_manager->applyShape(output_ind, new_shape);
  assert(output->buffer() != nullptr);
}

void DynamicShapeInferer::visit(const ir::operation::BatchMatMul &op)
{
  const auto lhs_index = op.getInputs().at(ir::operation::BatchMatMul::Input::LHS);
  const auto rhs_index = op.getInputs().at(ir::operation::BatchMatMul::Input::RHS);
  auto lhs = _tensor_registry->getITensor(lhs_index);
  auto rhs = _tensor_registry->getITensor(rhs_index);

  if (!lhs->is_dynamic() && !rhs->is_dynamic())
    return;

  const auto output_index = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_index);

  auto lhs_shape = lhs->getShape();
  auto rhs_shape = rhs->getShape();
  // TODO

  auto new_shape = shape_inference::inferBatchMatMulShape(lhs_shape, rhs_shape, op.param());
  _dynamic_tensor_manager->applyShape(output_index, new_shape);
}

void DynamicShapeInferer::visit(const ir::operation::BroadcastTo &op)
{
  auto output_ind = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_ind);

  auto input_idx = op.getInputs().at(ir::operation::BroadcastTo::INPUT);
  auto input = _tensor_registry->getITensor(input_idx);

  if ((!input->is_dynamic()) && (!output->is_dynamic()))
    return;

  auto shape_idx = op.getInputs().at(ir::operation::Tile::Input::MULTIPLES);
  const auto &shape = _tensor_registry->getITensor(shape_idx);

  assert(shape); // It shouldn't be 0.

  auto output_shape = shape_inference::inferBroadcastToShape(
      shape->getShape(), reinterpret_cast<const int32_t *>(shape->buffer()));

  // set output shape and output buffer
  _dynamic_tensor_manager->applyShape(output_ind, output_shape);
  assert(output->buffer() != nullptr);
}

void DynamicShapeInferer::visit(const ir::operation::Cast &op)
{
  handleSimpleUnaryOp(op, op.getInputs().at(ir::operation::Cast::INPUT));
}

void DynamicShapeInferer::visit(const ir::operation::Comparison &op)
{
  handleBinaryArithmeticOp(op, op.getInputs().at(ir::operation::Comparison::Input::INPUT0),
                           op.getInputs().at(ir::operation::Comparison::Input::INPUT1));
}

void DynamicShapeInferer::visit(const ir::operation::Concat &op)
{
  /*
    The state after compilation (satic shape inference) could be one of the following:

              inputs                  output        execution-time shape inf required
      ------------------------------------------    ---------------------------------
      case 1) all static              static         X
      case 2) at least on is dynamic  dynamic        O

    Then nnfw_apply_tensorinf() could change one or both inputs dynamic.
    So, in this method, we have one more state and we have to re-calculate shape for this shape.

      case 3) at least on is dynamic  static         O

    So, only when all inputs are static, we can skip dynamic shape inference.
  */
  bool all_static = true;
  for (auto input_ind : op.getInputs())
  {
    auto input = _tensor_registry->getITensor(input_ind);
    if (input->is_dynamic())
    {
      all_static = false;
      break;
    }
  }

  if (all_static)
    return;

  // sanity check
  {
    auto isConcatible = [](const backend::ITensor *input1, const backend::ITensor *input2,
                           int32_t axis) {
      if (input1->num_dimensions() != input2->num_dimensions())
        return false;

      for (size_t i = 0; i < input1->num_dimensions(); i++)
      {
        auto positive_axis = (axis >= 0) ? axis : axis + input1->num_dimensions();

        if (i != positive_axis)
          if (input1->dimension(i) != input2->dimension(i))
            return false;
      }

      return true;
    };

    auto first_input_ind = op.getInputs().at(0);
    auto first_input = _tensor_registry->getITensor(first_input_ind);

    for (auto input_ind : op.getInputs())
    {
      auto input = _tensor_registry->getITensor(input_ind);
      if (input != first_input && !isConcatible(first_input.get(), input.get(), op.param().axis))
        throw std::runtime_error("input shapes does not matched for concat");
    }
  }

  // getting output shape
  onert::shape_inference::Shapes in_shapes;
  for (auto input_ind : op.getInputs())
  {
    auto input = _tensor_registry->getITensor(input_ind);
    ir::Shape shape = input->getShape();

    in_shapes.emplace_back(shape);
  }

  auto output_ind = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_ind);
  auto output_shape = shape_inference::inferConcatShape(in_shapes, op.param());

  _dynamic_tensor_manager->applyShape(output_ind, output_shape);
}

void DynamicShapeInferer::visit(const ir::operation::Conv2D &op)
{
  // check if input is not dynamic
  auto input_ind = op.getInputs().at(ir::operation::Conv2D::INPUT);
  auto input = _tensor_registry->getITensor(input_ind);

  auto ker_ind = op.getInputs().at(ir::operation::Conv2D::KERNEL);
  auto ker = _tensor_registry->getITensor(ker_ind);

  if ((!input->is_dynamic()) && (!ker->is_dynamic()))
    return;

  ir::Shape input_shape = input->getShape();
  ir::Shape ker_shape = ker->getShape();

  auto output_ind = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_ind);

  ir::Shape output_shape = shape_inference::inferConv2DShape(input_shape, ker_shape, op.param());

  _dynamic_tensor_manager->applyShape(output_ind, output_shape);
  assert(output->buffer() != nullptr);
}

void DynamicShapeInferer::visit(const ir::operation::Cos &op)
{
  handleSimpleUnaryOp(op, op.getInputs().at(ir::operation::Cos::Input::INPUT));
}

void DynamicShapeInferer::visit(const ir::operation::Div &op)
{
  handleBinaryArithmeticOp(op, op.getInputs().at(ir::operation::Div::Input::LHS),
                           op.getInputs().at(ir::operation::Div::Input::RHS));
}

void DynamicShapeInferer::visit(const ir::operation::Exp &op)
{
  handleSimpleUnaryOp(op, op.getInputs().at(ir::operation::Exp::Input::INPUT));
}

void DynamicShapeInferer::visit(const ir::operation::ExpandDims &op)
{
  // check if input is not dynamic
  auto input_ind = op.getInputs().at(ir::operation::ExpandDims::INPUT);
  auto input = _tensor_registry->getITensor(input_ind);

  // check if output is not dynamic, meaning when 1st input is static and 2nd input is const
  auto output_ind = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_ind);

  /*
    Here, the state after compilation (satic shape inference) could be one of the following:

              input1     input2      output     execution-time shape inf required
              -----------------------------     --------------------------------
      case 1) static     const       static      X
      case 2) static    placeholder  dynamic     O
      case 3) dynamic    const       dynamic     O
      case 4) dynamic   placeholder  dynamic     O

    Then nnfw_apply_tensorinf() could change input dynamic.
    So, in this method, we could have one more state and we have to re-calculate shape
    for this shape.

      case 5) dynamic    const       static       O

    So, only when input1 and ouput are static, we can skip dynamic shape inference.
  */
  if ((!input->is_dynamic()) && (!output->is_dynamic()))
    return;

  ir::Shape input_shape = input->getShape();

  auto axis_ind = op.getInputs().at(ir::operation::ExpandDims::AXIS);
  auto axis = _tensor_registry->getITensor(axis_ind);
  auto axis_buf = reinterpret_cast<const int32_t *>(axis->buffer());
  assert(axis_buf);

  auto output_shape = shape_inference::inferExpandDimsShape(input_shape, axis_buf[0]);

  _dynamic_tensor_manager->applyShape(output_ind, output_shape);
  assert(output->buffer() != nullptr);
}

void DynamicShapeInferer::visit(const ir::operation::Fill &op)
{
  // check if output is not dynamic
  auto output_ind = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_ind);
  auto input_ind = op.getInputs().at(ir::operation::Fill::Input::INPUT);
  auto input = _tensor_registry->getITensor(input_ind);
  ir::Shape input_shape = input->getShape();

  if ((!input->is_dynamic()) && (!output->is_dynamic()))
    return;

  assert(input.get()->data_type() == ir::DataType::INT32);

  auto input_buf = reinterpret_cast<const int32_t *>(input->buffer());
  assert(input_buf);

  auto output_shape = shape_inference::inferFillShape(input_shape, input_buf);

  _dynamic_tensor_manager->applyShape(output_ind, output_shape);
  assert(output->buffer() != nullptr);
}

void DynamicShapeInferer::visit(const ir::operation::FullyConnected &op)
{
  const auto input_idx{op.getInputs().at(ir::operation::FullyConnected::Input::INPUT)};
  const auto &input = _tensor_registry->getITensor(input_idx);

  const auto ker_idx{op.getInputs().at(ir::operation::FullyConnected::Input::WEIGHT)};
  const auto &ker = _tensor_registry->getITensor(ker_idx);

  if (!input->is_dynamic() && !ker->is_dynamic())
    return;

  auto input_shape = input->getShape();
  auto ker_shape = ker->getShape();

  ir::Shape new_shape = shape_inference::inferFullyConnectedShape(input_shape, ker_shape);

  auto output_ind = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_ind);

  _dynamic_tensor_manager->applyShape(output_ind, new_shape);
  assert(output->buffer() != nullptr);
}

void DynamicShapeInferer::visit(const ir::operation::FusedBatchNorm &op)
{
  handleSimpleUnaryOp(op, op.getInputs().at(ir::operation::FusedBatchNorm::Input::INPUT));
}

void DynamicShapeInferer::visit(const ir::operation::Gather &op)
{
  const auto input_idx{op.getInputs().at(ir::operation::Gather::Input::INPUT)};
  const auto &input = _tensor_registry->getITensor(input_idx);
  auto input_shape = input->getShape();

  const auto indices_idx{op.getInputs().at(ir::operation::Gather::Input::INDICES)};
  const auto &indices = _tensor_registry->getITensor(indices_idx);
  auto indices_shape = indices->getShape();

  if (!(input->is_dynamic()) && !(indices->is_dynamic()))
    return;

  const auto rank = input_shape.rank();
  const auto axis = ((op.param().axis < 0) ? rank + op.param().axis : op.param().axis);

  assert(0 <= axis && axis < rank);

  ir::Shape new_shape = shape_inference::inferGatherShape(input_shape, indices_shape, axis, rank);

  auto output_ind = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_ind);

  _dynamic_tensor_manager->applyShape(output_ind, new_shape);
  assert(output->buffer() != nullptr);
}

void DynamicShapeInferer::visit(const ir::operation::Log &op)
{
  handleSimpleUnaryOp(op, op.getInputs().at(ir::operation::Log::Input::INPUT));
}

void DynamicShapeInferer::visit(const ir::operation::LogicalNot &op)
{
  handleSimpleUnaryOp(op, op.getInputs().at(ir::operation::LogicalNot::Input::INPUT));
}

void DynamicShapeInferer::visit(const ir::operation::LogicalOr &op)
{
  handleBinaryArithmeticOp(op, op.getInputs().at(ir::operation::LogicalOr::Input::INPUT0),
                           op.getInputs().at(ir::operation::LogicalOr::Input::INPUT1));
}

void DynamicShapeInferer::visit(const ir::operation::Logistic &op)
{
  handleSimpleUnaryOp(op, op.getInputs().at(ir::operation::Logistic::INPUT));
}

void DynamicShapeInferer::visit(const ir::operation::L2Normalization &op)
{
  handleSimpleUnaryOp(op, op.getInputs().at(ir::operation::L2Normalization::INPUT));
}

void DynamicShapeInferer::visit(const ir::operation::MatrixBandPart &op)
{
  handleSimpleUnaryOp(op, op.getInputs().at(ir::operation::MatrixBandPart::INPUT));
}

void DynamicShapeInferer::visit(const ir::operation::Max &op)
{
  handleBinaryArithmeticOp(op, op.getInputs().at(ir::operation::Max::Input::LHS),
                           op.getInputs().at(ir::operation::Max::Input::RHS));
}

void DynamicShapeInferer::visit(const ir::operation::Min &op)
{
  handleBinaryArithmeticOp(op, op.getInputs().at(ir::operation::Min::Input::LHS),
                           op.getInputs().at(ir::operation::Min::Input::RHS));
}

void DynamicShapeInferer::visit(const ir::operation::Mul &op)
{
  handleBinaryArithmeticOp(op, op.getInputs().at(ir::operation::Mul::Input::LHS),
                           op.getInputs().at(ir::operation::Mul::Input::RHS));
}

void DynamicShapeInferer::visit(const ir::operation::Neg &op)
{
  handleSimpleUnaryOp(op, op.getInputs().at(ir::operation::Neg::Input::INPUT));
}

void DynamicShapeInferer::visit(const ir::operation::OneHot &op)
{
  auto output_ind = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_ind);

  auto indices_ind = op.getInputs().at(ir::operation::OneHot::INDICES);
  const auto &indices = _tensor_registry->getITensor(indices_ind);
  auto indices_shape = indices->getShape();

  auto depth_ind = op.getInputs().at(ir::operation::OneHot::DEPTH);
  const auto &depth = _tensor_registry->getITensor(depth_ind);

  if (!indices->is_dynamic() && !depth->is_dynamic())
  {
    return;
  }

  int32_t *depth_buf = reinterpret_cast<int32_t *>(depth->buffer());
  assert(depth_buf);
  const auto axis_val = op.param().axis;

  ir::Shape new_shape = shape_inference::inferOnehotShape(indices_shape, *depth_buf, axis_val);
  _dynamic_tensor_manager->applyShape(output_ind, new_shape);
  assert(output->buffer() != nullptr);
}

void DynamicShapeInferer::visit(const ir::operation::Pack &op)
{
  bool is_any_of_inputs_dynamic = [&]() -> bool {
    for (uint32_t i = 0; i < op.getInputs().size(); ++i)
    {
      const auto &input = _tensor_registry->getITensor(op.getInputs().at(i));
      if (input->is_dynamic())
      {
        return true;
      }
    }
    return false;
  }();

  const auto input_idx{op.getInputs().at(0)};
  const auto &input = _tensor_registry->getITensor(input_idx);
  auto input_shape = input->getShape();

  auto output_ind = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_ind);

  if (!is_any_of_inputs_dynamic && !output->is_dynamic())
    return;

  const auto rank = input_shape.rank() + 1;
  const auto axis = ((op.param().axis < 0) ? rank + op.param().axis : op.param().axis);
  const auto num = op.param().num;

  assert(0 <= axis && axis < rank);

  ir::Shape new_shape = shape_inference::inferPackShape(input_shape, axis, rank, num);

  _dynamic_tensor_manager->applyShape(output_ind, new_shape);
  assert(output->buffer() != nullptr);
}

void DynamicShapeInferer::visit(const ir::operation::Pad &op)
{
  // check if output is not dynamic
  auto output_ind = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_ind);

  auto input_ind = op.getInputs().at(ir::operation::Pad::Input::INPUT);
  auto input = _tensor_registry->getITensor(input_ind);

  auto pad_ind = op.getInputs().at(ir::operation::Pad::Input::PAD);
  auto pad = _tensor_registry->getITensor(pad_ind);

  // check if input and output are not dynamic
  if ((!input->is_dynamic()) && (!output->is_dynamic()))
    return;

  int32_t *pad_buf = reinterpret_cast<int32_t *>(pad->buffer());
  assert(pad_buf);

  auto output_shape =
      shape_inference::inferPadShape(input->getShape(), pad_buf, pad->getShape().num_elements());

  // change output shape and reallocate output tensor memory
  _dynamic_tensor_manager->applyShape(output_ind, output_shape);
  assert(output->buffer() != nullptr);
}

void DynamicShapeInferer::visit(const ir::operation::Permute & /* op */)
{
  // NOTE Permute is a special operation which does not do shape inference before the actual
  // function(kernel) execution. Shape inference and output allocation will be done in the kernel
  // on-the-fly, as it must support inter-backend inference/allocation.
}

void DynamicShapeInferer::visit(const ir::operation::Pow &op)
{
  handleBinaryArithmeticOp(op, op.getInputs().at(ir::operation::Pow::Input::LHS),
                           op.getInputs().at(ir::operation::Pow::Input::RHS));
}

void DynamicShapeInferer::visit(const ir::operation::Range &op)
{
  // check if output is not dynamic
  auto output_ind = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_ind);

  // from op, access the buffer of second input to read new shape
  auto start_idx = op.getInputs().at(ir::operation::Range::Input::START);
  auto start_tensor = _tensor_registry->getITensor(start_idx);

  auto limit_idx = op.getInputs().at(ir::operation::Range::Input::LIMIT);
  auto limit_tensor = _tensor_registry->getITensor(limit_idx);

  auto delta_idx = op.getInputs().at(ir::operation::Range::Input::DELTA);
  auto delta_tensor = _tensor_registry->getITensor(delta_idx);

  if (!start_tensor->is_dynamic() && !limit_tensor->is_dynamic() && !delta_tensor->is_dynamic() &&
      !output->is_dynamic())
    return;

  ir::Shape new_shape;
  if (output->data_type() == ir::DataType::FLOAT32)
  {
    new_shape =
        shape_inference::inferRangeShape<float>(*reinterpret_cast<float *>(start_tensor->buffer()),
                                                *reinterpret_cast<float *>(limit_tensor->buffer()),
                                                *reinterpret_cast<float *>(delta_tensor->buffer()));
  }
  else if (output->data_type() == ir::DataType::INT32)
  {
    new_shape = shape_inference::inferRangeShape<int32_t>(
        *reinterpret_cast<int32_t *>(start_tensor->buffer()),
        *reinterpret_cast<int32_t *>(limit_tensor->buffer()),
        *reinterpret_cast<int32_t *>(delta_tensor->buffer()));
  }
  _dynamic_tensor_manager->applyShape(output_ind, new_shape);
  assert(output->buffer() != nullptr);
}

void DynamicShapeInferer::visit(const ir::operation::Reduce &op)
{
  const auto input_idx{op.getInputs().at(ir::operation::Reduce::Input::INPUT)};
  const auto &input = _tensor_registry->getITensor(input_idx);
  auto input_shape = input->getShape();

  const auto axes_idx{op.getInputs().at(ir::operation::Reduce::Input::AXES)};
  const auto &axes = _tensor_registry->getITensor(axes_idx);

  if (!input->is_dynamic())
    return;

  std::vector<int32_t> axes_vec;
  for (uint32_t i = 0; i < axes->getShape().num_elements(); ++i)
  {
    const auto buffer = axes->buffer() + axes->calcOffset({i});
    switch (axes->data_type())
    {
      case ir::DataType::INT32:
      {
        axes_vec.emplace_back(*reinterpret_cast<const int32_t *>(buffer));
        break;
      }
      case ir::DataType::INT64:
      {
        axes_vec.emplace_back(*reinterpret_cast<const int64_t *>(buffer));
        break;
      }
      default:
        throw std::runtime_error("DynamicShapeInferer " + op.name() + ": Not supported data type");
        break;
    }
  }
  const auto keep_dims = op.param().keep_dims;

  auto output_ind = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_ind);

  ir::Shape new_shape = shape_inference::inferReduceShape(input_shape, axes_vec, keep_dims);

  _dynamic_tensor_manager->applyShape(output_ind, new_shape);
  assert(output->buffer() != nullptr);
}

void DynamicShapeInferer::visit(const ir::operation::Reshape &op)
{
  // check if output is not dynamic
  auto output_ind = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_ind);

  auto input_ind = op.getInputs().at(ir::operation::Reshape::Input::INPUT);
  auto input = _tensor_registry->getITensor(input_ind);

  /*
    Here, the state after compilation (satic shape inference) could be one of the following:

              input1   input2 (or option)   output     execution-time shape inf required
              ------------------------------------     --------------------------------
      case 1) static         const          static       X
      case 2) static      placeholder       dynamic      O
      case 3) dynamic        const          dynamic      O
      case 4) dynamic     placeholder       dynamic      O

    Then nnfw_apply_tensorinf() could change input dynamic.
    So, in this method, we could have one more state and we have to re-calculate shape
    for this shape.

      case 5) dynamic    const       static       O

    So, only when both input1 and ouput are static, we can skip dynamic shape inference.
  */
  if ((!input->is_dynamic()) && (!output->is_dynamic()))
    return;

  // New shape is given by second input tensor
  if (op.getInputs().size() == 2)
  {
    // from op, access the buffer of second input to read new shape
    auto new_shape_ind = op.getInputs().at(ir::operation::Reshape::Input::SHAPE);

    // getting output shape by reading new_shape tensor buffer
    auto new_shape = _tensor_registry->getITensor(new_shape_ind);
    assert(new_shape);

    int32_t *new_shape_buf = reinterpret_cast<int32_t *>(new_shape->buffer());
    assert(new_shape_buf);

    auto output_shape = shape_inference::inferReshapeShape(
        new_shape_buf, new_shape->getShape().num_elements(), input->getShape().num_elements());

    // if shape is changed, change output shape and reallocate output tensor memory
    if (output_shape != output->getShape() || output->buffer() == nullptr)
    {
      // change on output shape
      _dynamic_tensor_manager->applyShape(output_ind, output_shape);
    }
    assert(output->buffer() != nullptr);
  }
  // New shape is given by option
  else if (op.param().new_shape.size() != 0)
  {
    // Let's check the new_shape option
    auto shape = op.param().new_shape;
    auto output_shape = shape_inference::inferReshapeShape(shape.data(), shape.size(),
                                                           input->getShape().num_elements());

    // if shape is changed, change output shape and reallocate output tensor memory
    if (output_shape != output->getShape() || output->buffer() == nullptr)
    {
      // change on output shape
      _dynamic_tensor_manager->applyShape(output_ind, output_shape);
    }
    assert(output->buffer() != nullptr);
  }
  else
  {
    throw std::runtime_error("Reshape: new shape is missing");
    return;
  }
}

void DynamicShapeInferer::visit(const ir::operation::Reverse &op)
{
  handleSimpleUnaryOp(op, op.getInputs().at(ir::operation::Reverse::INPUT));
}

void DynamicShapeInferer::visit(const ir::operation::Round &op)
{
  handleSimpleUnaryOp(op, op.getInputs().at(ir::operation::Round::Input::INPUT));
}

void DynamicShapeInferer::visit(const ir::operation::RSQRT &op)
{
  handleSimpleUnaryOp(op, op.getInputs().at(ir::operation::RSQRT::INPUT));
}

void DynamicShapeInferer::visit(const ir::operation::Select &op)
{
  const auto input_cond_idx = op.getInputs().at(ir::operation::Select::Input::CONDITION);
  const auto &input_cond = _tensor_registry->getITensor(input_cond_idx);

  const auto input_true_idx = op.getInputs().at(ir::operation::Select::Input::INPUT_TRUE);
  const auto &input_true = _tensor_registry->getITensor(input_true_idx);

  const auto input_false_idx = op.getInputs().at(ir::operation::Select::Input::INPUT_FALSE);
  const auto &input_false = _tensor_registry->getITensor(input_false_idx);

  if ((!input_cond->is_dynamic()) && (!input_true->is_dynamic()) && (!input_false->is_dynamic()))
  {
    return;
  }

  auto input_cond_shape = input_cond->getShape();
  auto input_true_shape = input_true->getShape();
  auto input_false_shape = input_false->getShape();

  // Select output shpae
  ir::Shape new_shape =
      shape_inference::inferSelectShape(input_cond_shape, input_true_shape, input_false_shape);

  auto output_ind = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_ind);

  _dynamic_tensor_manager->applyShape(output_ind, new_shape);
  assert(output->buffer() != nullptr);
}

void DynamicShapeInferer::visit(const ir::operation::Shape &op)
{
  const auto input_idx{op.getInputs().at(0)};
  const auto &input = _tensor_registry->getITensor(input_idx);
  auto input_shape = input->getShape();

  if (!input->is_dynamic())
    return;

  auto output_ind = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_ind);

  ir::Shape output_shape;
  output_shape.append(input_shape.rank());

  _dynamic_tensor_manager->applyShape(output_ind, output_shape);
  assert(output->buffer() != nullptr);
}

void DynamicShapeInferer::visit(const ir::operation::Sin &op)
{
  handleSimpleUnaryOp(op, op.getInputs().at(ir::operation::Sin::Input::INPUT));
}

void DynamicShapeInferer::visit(const ir::operation::Slice &op)
{
  const auto input_index{op.getInputs().at(ir::operation::Slice::Input::INPUT)};
  const auto input = _tensor_registry->getITensor(input_index);
  const auto begins_index{op.getInputs().at(ir::operation::Slice::Input::BEGINS)};
  const auto begins = _tensor_registry->getITensor(begins_index);
  const auto sizes_index{op.getInputs().at(ir::operation::Slice::Input::SIZES)};
  const auto sizes = _tensor_registry->getITensor(sizes_index);
  auto output_index = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_index);

  if (!(input->is_dynamic() || begins->is_dynamic() || sizes->is_dynamic() || output->is_dynamic()))
  {
    return;
  }

  ir::Shape input_shape = input->getShape();
  auto begins_buf = reinterpret_cast<const int32_t *>(begins->buffer());
  auto sizes_buf = reinterpret_cast<const int32_t *>(sizes->buffer());

  ir::Shape new_shape = shape_inference::inferSliceShape(input_shape, begins_buf, sizes_buf);

  _dynamic_tensor_manager->applyShape(output_index, new_shape);
  assert(output->buffer() != nullptr);
}

void DynamicShapeInferer::visit(const ir::operation::Softmax &op)
{
  handleSimpleUnaryOp(op, op.getInputs().at(ir::operation::Softmax::INPUT));
}

void DynamicShapeInferer::visit(const ir::operation::SpaceToBatchND &op)
{
  const auto input_idx{op.getInputs().at(ir::operation::SpaceToBatchND::Input::INPUT)};
  const auto block_shape_idx{op.getInputs().at(ir::operation::SpaceToBatchND::Input::BLOCK_SIZE)};
  const auto padding_idx{op.getInputs().at(ir::operation::SpaceToBatchND::Input::PADDINGS)};
  auto output_idx{op.getOutputs().at(0)};

  const auto &input = _tensor_registry->getITensor(input_idx);
  const auto &block_shape = _tensor_registry->getITensor(block_shape_idx);
  const auto &padding = _tensor_registry->getITensor(padding_idx);
  auto output = _tensor_registry->getITensor(output_idx);

  if (!(input->is_dynamic() || block_shape->is_dynamic() || padding->is_dynamic() ||
        output->is_dynamic()))
  {
    return;
  }

  auto input_shape = input->getShape();
  auto block_shape_shape = block_shape->getShape();
  auto padding_shape = padding->getShape();

  auto block_shape_data = reinterpret_cast<int32_t *>(block_shape->buffer());
  auto padding_data = reinterpret_cast<int32_t *>(padding->buffer());

  ir::Shape new_shape = shape_inference::inferSpaceToBatchNDShape(
      input_shape, block_shape_shape, padding_shape, block_shape_data, padding_data);

  _dynamic_tensor_manager->applyShape(output_idx, new_shape);
  assert(output->buffer() != nullptr);
}

void DynamicShapeInferer::visit(const ir::operation::Split &op)
{
  const auto input_idx{op.getInputs().at(ir::operation::Split::Input::INPUT)};
  const auto &input = _tensor_registry->getITensor(input_idx);

  if (!input->is_dynamic())
  {
    return;
  }

  auto input_shape = input->getShape();

  const auto axis = op.param().axis;
  const auto num_splits = op.param().num_splits;
  const auto rank = input_shape.rank();
  auto axis_resolved = axis < 0 ? axis + rank : axis;

  assert(0 <= axis_resolved && axis_resolved < rank);

  ir::Shape new_shape = shape_inference::inferSplitShape(input_shape, axis_resolved, num_splits);
  for (int out_tensor_idx = 0; out_tensor_idx < num_splits; out_tensor_idx++)
  {
    auto output_ind = op.getOutputs().at(out_tensor_idx);
    auto output = _tensor_registry->getITensor(output_ind);

    _dynamic_tensor_manager->applyShape(output_ind, new_shape);
    assert(output->buffer() != nullptr);
  }
}

void DynamicShapeInferer::visit(const ir::operation::SquaredDifference &op)
{
  handleBinaryArithmeticOp(op, op.getInputs().at(ir::operation::SquaredDifference::Input::LHS),
                           op.getInputs().at(ir::operation::SquaredDifference::Input::RHS));
}

void DynamicShapeInferer::visit(const ir::operation::Squeeze &op)
{
  const auto input_idx{op.getInputs().at(ir::operation::Squeeze::Input::INPUT)};
  const auto &input = _tensor_registry->getITensor(input_idx);

  if (!input->is_dynamic())
  {
    return;
  }

  auto input_shape = input->getShape();

  // Squeeze output shpae
  ir::Shape new_shape = shape_inference::inferSqueezeShape(input_shape, op.param());

  auto output_ind = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_ind);

  _dynamic_tensor_manager->applyShape(output_ind, new_shape);
  assert(output->buffer() != nullptr);
}

void DynamicShapeInferer::visit(const ir::operation::StridedSlice &op)
{

  const auto input_index{op.getInputs().at(ir::operation::StridedSlice::Input::INPUT)};
  auto input = _tensor_registry->getITensor(input_index);
  ir::Shape input_shape = input->getShape();

  const auto starts_index{op.getInputs().at(ir::operation::StridedSlice::Input::STARTS)};
  auto starts = _tensor_registry->getITensor(starts_index);

  const auto ends_index{op.getInputs().at(ir::operation::StridedSlice::Input::ENDS)};
  auto ends = _tensor_registry->getITensor(ends_index);

  const auto strides_index{op.getInputs().at(ir::operation::StridedSlice::Input::STRIDES)};
  auto strides = _tensor_registry->getITensor(strides_index);

  if (!(input->is_dynamic() || starts->is_dynamic() || ends->is_dynamic() || strides->is_dynamic()))
  {
    return;
  }

  const auto begin_mask = op.param().begin_mask;
  const auto end_mask = op.param().end_mask;
  const auto shrink_axis_mask = op.param().shrink_axis_mask;
  const auto rank = input_shape.rank();

  auto op_params = shape_inference::buildStridedSliceParams(
      reinterpret_cast<uint32_t *>(starts->buffer()), reinterpret_cast<uint32_t *>(ends->buffer()),
      reinterpret_cast<uint32_t *>(strides->buffer()), begin_mask, end_mask, shrink_axis_mask,
      rank);

  auto output_index = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_index);

  ir::Shape output_shape =
      onert::shape_inference::inferStridedSliceShape(input_shape, op_params, rank);

  _dynamic_tensor_manager->applyShape(output_index, output_shape);
  assert(output->buffer() != nullptr);
}

void DynamicShapeInferer::visit(const ir::operation::Sub &op)
{
  handleBinaryArithmeticOp(op, op.getInputs().at(ir::operation::Sub::Input::LHS),
                           op.getInputs().at(ir::operation::Sub::Input::RHS));
}

void DynamicShapeInferer::visit(const ir::operation::Tanh &op)
{
  handleSimpleUnaryOp(op, op.getInputs().at(ir::operation::Tanh::INPUT));
}

void DynamicShapeInferer::visit(const ir::operation::Tile &op)
{
  auto output_ind = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_ind);

  auto input_idx = op.getInputs().at(ir::operation::Tile::Input::INPUT);
  auto input = _tensor_registry->getITensor(input_idx);

  auto multiplier_idx = op.getInputs().at(ir::operation::Tile::Input::MULTIPLES);
  auto multiplier = _tensor_registry->getITensor(multiplier_idx);

  if ((!input->is_dynamic()) && (!output->is_dynamic()))
    return;

  auto input_shape = input->getShape();
  auto multiplier_buffer = reinterpret_cast<const int32_t *>(multiplier->buffer());
  assert(multiplier_buffer);

  auto output_shape = shape_inference::inferTileShape(input_shape, multiplier_buffer);

  // set output shape and output buffer
  _dynamic_tensor_manager->applyShape(output_ind, output_shape);
  assert(output->buffer() != nullptr);
}

void DynamicShapeInferer::visit(const ir::operation::Transpose &op)
{
  // check if output is not dynamic
  auto output_ind = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_ind);

  // from op, access the buffer of second input to read new shape
  auto input_ind = op.getInputs().at(ir::operation::Transpose::Input::INPUT);
  auto input_tensor = _tensor_registry->getITensor(input_ind);
  auto input_shape = input_tensor->getShape();

  if (!input_tensor->is_dynamic())
    return;

  const auto perm{op.param().perm};
  // set output shape, based on input and params
  ir::Shape new_shape = shape_inference::inferTransposeShape(input_shape, perm);

  _dynamic_tensor_manager->applyShape(output_ind, new_shape);
  assert(output->buffer() != nullptr);
}

void DynamicShapeInferer::visit(const ir::operation::Unpack &op)
{
  // check if output is not dynamic
  const auto input_idx{op.getInputs().at(0)};
  const auto &input = _tensor_registry->getITensor(input_idx);

  if (!input->is_dynamic())
    return;

  auto input_shape = input->getShape();

  const auto rank = input_shape.rank();
  const auto axis = ((op.param().axis < 0) ? rank + op.param().axis : op.param().axis);
  const auto num = op.param().num;

  assert(0 <= axis && axis < rank);

  ir::Shape new_shape = shape_inference::inferUnpackShape(input_shape, axis, rank);

  for (int out_tensor_idx = 0; out_tensor_idx < num; out_tensor_idx++)
  {
    auto output_ind = op.getOutputs().at(out_tensor_idx);
    auto output = _tensor_registry->getITensor(output_ind);

    _dynamic_tensor_manager->applyShape(output_ind, new_shape);

    assert(output->buffer() != nullptr);
  }
}

void DynamicShapeInferer::visit(const ir::operation::ZerosLike &op)
{
  handleSimpleUnaryOp(op, op.getInputs().at(ir::operation::ZerosLike::INPUT));
}

} // namespace exec
} // namespace onert
