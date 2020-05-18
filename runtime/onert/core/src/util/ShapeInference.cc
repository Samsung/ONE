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

#include "util/Utils.h"
#include "ir/InternalType.h"
#include "ir/Shape.h"
#include "ir/operation/AvgPool2D.h"
#include "ir/operation/MaxPool2D.h"
#include "util/ShapeInference.h"

namespace onert
{
namespace shape_inference
{

//
// Helper functions
//

namespace
{

template <typename T, typename U>
typename std::enable_if<std::is_integral<T>::value && std::is_integral<U>::value,
                        typename std::common_type<T, U>::type>::type
ceil_div(T dividend, U divisor)
{
  assert(dividend > 0 && divisor > 0 && "this implementations is for positive numbers only");
  return (dividend + divisor - 1) / divisor;
}

// Calculate the result of broadcast of two shapes
ir::Shape broadcastShapes(const ir::Shape &lhs_shape, const ir::Shape &rhs_shape)
{
  ir::Shape out_shape;
  auto max_rank = std::max(lhs_shape.rank(), rhs_shape.rank());

  for (int idx = 0; idx < max_rank; ++idx)
  {
    // Go over operands dimensions from right to left
    int lhs_idx = lhs_shape.rank() - idx - 1;
    int rhs_idx = rhs_shape.rank() - idx - 1;

    int32_t lhs_dim = lhs_idx >= 0 ? lhs_shape.dim(lhs_idx) : 1;
    int32_t rhs_dim = rhs_idx >= 0 ? rhs_shape.dim(rhs_idx) : 1;

    if (lhs_dim != 1 && rhs_dim != 1 && lhs_dim != rhs_dim)
      throw std::runtime_error("Incompatible shapes for broadcast");

    out_shape.prepend(std::max(lhs_dim, rhs_dim));
  }

  return out_shape;
}

// Calculate output height and width of convolution-like operation
std::pair<int, int> calcConvLikeHeightAndWidth(const int in_h, const int in_w, const int ker_h,
                                               const int ker_w, const ir::Padding pad,
                                               const ir::Stride stride)
{
  int32_t out_h = 0, out_w = 0;

  switch (pad.type)
  {
    case ir::PaddingType::SAME:
      out_h = ceil_div(in_h, stride.vertical);
      out_w = ceil_div(in_w, stride.horizontal);
      break;
    case ir::PaddingType::VALID:
      out_h = ceil_div(in_h - ker_h + 1, stride.vertical);
      out_w = ceil_div(in_w - ker_w + 1, stride.horizontal);
      break;
    case ir::PaddingType::EXPLICIT:
      out_h = (in_h + pad.param.top + pad.param.bottom - ker_h) / stride.vertical + 1;
      out_w = (in_w + pad.param.left + pad.param.right - ker_w) / stride.horizontal + 1;
      break;
    default:
      assert(false);
  }

  return {out_h, out_w};
}

} // namespace

//
// Shape inference
//

// Define shape calculation for operations. List them in alphabetic order.
// Remove TODO when the function name matching the alphabet is added

Shapes inferEltwiseShape(const ir::Shape &lhs_shape, const ir::Shape &rhs_shape)
{
  return {broadcastShapes(lhs_shape, rhs_shape)};
}

Shapes inferAvgPoolShape(const ir::Shape &in_shape, const ir::operation::AvgPool2D::Param &param,
                         const ir::Layout layout)
{
  assert(layout == ir::Layout::NHWC);
  auto ifm_shape = in_shape.asFeature(layout);
  const auto out_h_w = calcConvLikeHeightAndWidth(ifm_shape.H, ifm_shape.W, param.kh, param.kw,
                                                  param.padding, param.stride);
  // Pooling don't change number of channels and batch size
  return {ir::Shape{ifm_shape.N, out_h_w.first, out_h_w.second, ifm_shape.C}};
}

ir::Shape inferConcatShape(const Shapes &in_shapes, const ir::operation::Concat::Param &param)
{
  const int32_t concat_axis = param.axis;
  const auto &first_in_shape = in_shapes[0];

  // Check that all shapes are equal except for concat axis dimension
  for (const auto &in_shape : in_shapes)
  {
    assert(in_shape.rank() == first_in_shape.rank());
    for (int64_t dim_idx = 0; dim_idx < in_shape.rank(); ++dim_idx)
      assert(dim_idx == concat_axis || in_shape.dim(dim_idx) == first_in_shape.dim(dim_idx));
  }

  // Calculate output shape
  ir::Shape out_shape(first_in_shape);
  out_shape.dim(concat_axis) = 0;
  for (const auto &in_shape : in_shapes)
    out_shape.dim(concat_axis) += in_shape.dim(concat_axis);
  return out_shape;
}

Shapes inferConv2DShape(const ir::Shape &in_shape, const ir::Shape &ker_shape,
                        const ir::operation::Conv2D::Param &param, ir::Layout layout)
{
  assert(layout == ir::Layout::NHWC);
  auto ifm_shape = in_shape.asFeature(layout);

  // Kernel format is [depth_out, kernel_height, kernel_width, depth_in]
  auto kf_shape = ker_shape.asFeature(layout);
  assert(ifm_shape.C == kf_shape.C);

  const auto out_h_w = calcConvLikeHeightAndWidth(ifm_shape.H, ifm_shape.W, kf_shape.H, kf_shape.W,
                                                  param.padding, param.stride);

  return {ir::Shape{ifm_shape.N, out_h_w.first, out_h_w.second, kf_shape.N}};
}

Shapes inferDepthwiseConv2DShape(const ir::Shape &in_shape, const ir::Shape &ker_shape,
                                 const ir::operation::DepthwiseConv2D::Param &param,
                                 ir::Layout layout)
{
  assert(layout == ir::Layout::NHWC);
  auto ifm_shape = in_shape.asFeature(layout);

  // Kernel format is [1, kernel_height, kernel_width, depth_out]
  auto kf_shape = ker_shape.asFeature(layout);
  assert(kf_shape.C == static_cast<int32_t>(ifm_shape.C * param.multiplier));
  assert(kf_shape.N == 1);

  const auto out_h_w = calcConvLikeHeightAndWidth(ifm_shape.H, ifm_shape.W, kf_shape.H, kf_shape.W,
                                                  param.padding, param.stride);

  return {ir::Shape{ifm_shape.N, out_h_w.first, out_h_w.second, kf_shape.C}};
}

Shapes inferFullyConnectedShape(const ir::Shape &in_shape, const ir::Shape &ker_shape)
{
  assert(in_shape.rank() >= 2);
  assert(ker_shape.rank() == 2);

  const auto input_size_with_batch = in_shape.num_elements();
  const auto num_units = ker_shape.dim(0);
  const auto input_size = ker_shape.dim(1);
  const auto batch_size = input_size_with_batch / input_size;
  assert(input_size_with_batch % input_size == 0);

  return {{ir::Shape({static_cast<int32_t>(batch_size), num_units})}};
}

// TODO write op starting from G
// TODO write op starting from L

Shapes inferMaxPoolShape(const ir::Shape &in_shape, const ir::operation::MaxPool2D::Param &param,
                         const ir::Layout layout)
{
  assert(layout == ir::Layout::NHWC);
  auto ifm_shape = in_shape.asFeature(layout);
  const auto out_h_w = calcConvLikeHeightAndWidth(ifm_shape.H, ifm_shape.W, param.kh, param.kw,
                                                  param.padding, param.stride);
  // Pooling don't change number of channels and batch size
  return {ir::Shape{ifm_shape.N, out_h_w.first, out_h_w.second, ifm_shape.C}};
}

// TODO write op starting from N
// TODO write op starting from P
// TODO write op starting from R
// TODO write op starting from S
// TODO write op starting from T
// TODO write op starting from U
// TODO write op starting from Z

/*
  StaticInferer

  - Define visitors for operations. List them in alphabetic order.
  - Remove TODO when any op starting from the alphabet is added
*/

void StaticInferer::visit(const ir::operation::Add &op)
{
  const auto lhs_idx{op.getInputs().at(ir::operation::Add::Input::LHS)};
  const auto &lhs = _operands.at(lhs_idx);
  const auto rhs_idx{op.getInputs().at(ir::operation::Add::Input::RHS)};
  const auto &rhs = _operands.at(rhs_idx);
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

  if (lhs.info().memAllocType() == ir::MemAllocType::DYNAMIC ||
      rhs.info().memAllocType() == ir::MemAllocType::DYNAMIC)
  {
    output.info().memAllocType(ir::MemAllocType::DYNAMIC);
    return;
  }

  // re-sizing output shape
  ir::Shape new_shape = broadcastShapes(lhs.info().shape(), rhs.info().shape());
  output.info().shape(new_shape);
}

void StaticInferer::visit(const ir::operation::Concat &op)
{
  const auto input_count = op.getInputs().size();

  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

  Shapes input_shapes;
  for (uint32_t i = 0; i < input_count; i++)
  {
    const auto input_idx{op.getInputs().at(i)};
    const auto &input = _operands.at(input_idx);

    if (input.info().memAllocType() == ir::MemAllocType::DYNAMIC)
    {
      output.info().memAllocType(ir::MemAllocType::DYNAMIC);
      return;
    }

    input_shapes.emplace_back(input.shape());
  }

  ir::Shape out_shape = inferConcatShape(input_shapes, op.param());

  // re-sizing output shape
  output.info().shape(out_shape);
}

// TODO write op starting from D
// TODO write op starting from E
// TODO write op starting from F
// TODO write op starting from G
// TODO write op starting from L
// TODO write op starting from M
// TODO write op starting from N
// TODO write op starting from P

void StaticInferer::visit(const ir::operation::Reshape &op)
{
  const auto input_idx{op.getInputs().at(ir::operation::Reshape::Input::INPUT)};
  const auto &input = _operands.at(input_idx);

  // get mutable output operand
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

  // if input is dynamic, output also becomes dynamic
  if (input.info().memAllocType() == ir::MemAllocType::DYNAMIC)
  {
    output.info().memAllocType(ir::MemAllocType::DYNAMIC);
    return;
  }

  if (op.getInputs().size() == 1)
  {
    // no change on output shape
    return;
  }

  // Let's check the second input
  const auto shape_idx{op.getInputs().at(ir::operation::Reshape::Input::SHAPE)};
  const auto &shape = _operands.at(shape_idx);

  if (shape.isConstant())
  {
    // if shape is from Const, TFLC put the shape of output into tensor
    // no change on output shape
    return;
  }

  // if shape is NOT Const, set output shape to be dynamic_
  output.info().memAllocType(ir::MemAllocType::DYNAMIC);
}

// TODO write op starting from S
// TODO write op starting from T
// TODO write op starting from U
// TODO write op starting from Z

} // namespace shape_inference
} // namespace onert

// namespace for helper function of DynamicInferer
namespace
{

using onert::backend::ITensor;

bool isReshapableShape(const ITensor *input, const onert::ir::Shape &shape)
{
  size_t input_elem_conut = 1;
  {
    for (size_t axis = 0; axis < input->num_dimensions(); axis++)
      input_elem_conut *= input->dimension(axis);
  }

  return (input_elem_conut == shape.num_elements());
}

} // namespace

namespace onert
{
namespace shape_inference
{

/*
 * DynamicInferer

  - Define visitors for operations. List them in alphabetic order.
  - Remove TODO when any op starting from the alphabet is added
 */

// TODO write op starting from A
// TODO write op starting from C
// TODO write op starting from D
// TODO write op starting from E
// TODO write op starting from F
// TODO write op starting from G
// TODO write op starting from L
// TODO write op starting from M
// TODO write op starting from N
// TODO write op starting from P

void DynamicInferer::visit(const ir::operation::Reshape &op)
{
  // check if output is not dynamic
  auto output_ind = op.getOutputs().at(0);
  auto *output = _tensor_registry->getITensor(output_ind);
  if (!output->is_dynamic())
    return;

  // from op, access the buffer of second input to read new shape
  auto new_shape_ind = op.getInputs().at(ir::operation::Reshape::Input::SHAPE);
  auto &new_shape_op = _operands.at(new_shape_ind);

  // if shape is from Const, TFLC put the shape of output into tensor
  if (new_shape_op.isConstant())
  {
    // no change on output shape
    return;
  }

  // getting output shape by reading new_shape tensor buffer
  auto new_shape = _tensor_registry->getITensor(new_shape_ind);
  assert(new_shape);

  int32_t *new_shape_buf = reinterpret_cast<int32_t *>(new_shape->buffer());
  assert(new_shape_buf);

  auto new_rank = new_shape->dimension(0);

  ir::Shape output_shape(new_rank);
  for (size_t d = 0; d < new_rank; d++)
    output_shape.dim(d) = new_shape_buf[d];

  // sanity check
  {
    auto input_ind = op.getInputs().at(ir::operation::Reshape::Input::INPUT);
    auto input = _tensor_registry->getITensor(input_ind);
    assert(input);

    if (!isReshapableShape(input, output_shape))
      throw std::runtime_error("Reshape: 2nd param is not compatible with the shape of input");
  }

  // set output shape and output buffer
  setShape(output, output_shape);

  assert(output->buffer() == nullptr);
  _dynamic_tensor_manager->allocate(output_ind, output_shape);
  assert(output->buffer() != nullptr);
}

// TODO write op starting from S
// TODO write op starting from T
// TODO write op starting from U
// TODO write op starting from Z

} // namespace shape_inference
} // namespace onert
