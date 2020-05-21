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
#include "util/logging.h"

#include <sstream>

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

ir::Shape inferEltwiseShape(const ir::Shape &lhs_shape, const ir::Shape &rhs_shape)
{
  return broadcastShapes(lhs_shape, rhs_shape);
}

// TODO move this when Avgpool.cc is created in util/shapeinf
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

// TODO move this when Concat.cc is created in util/shapeinf
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

// TODO move this when Conv2D.cc is created in util/shapeinf
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

// TODO move this when DepthwiseConv2D.cc is created in util/shapeinf
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

// TODO move this when FullyConnected.cc is created in util/shapeinf
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

// TODO move this when MaxPool.cc is created in util/shapeinf
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

/*
  StaticInferer
  - Write methods except visit()
  - For visit() of each operator, find each op's C file
*/

void StaticInferer::handleSimpleUnaryOp(const ir::Operation &op, const ir::OperandIndex input_idx)
{
  const auto &input = _operands.at(input_idx);

  // get mutable output operand
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

  // if input is dynamic, output also becomes dynamic
  if (input.info().isDynamic())
  {
    output.info().setDynamic();
    return;
  }

  // re-sizing output shape
  ir::Shape new_shape = input.info().shape();
  output.info().shape(new_shape);
}

void StaticInferer::dump()
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

  _operands.iterate([&](const ir::OperandIndex &ind, const ir::Operand &operand) {
    VERBOSE(StaticInferer) << "Operand #" << ind.value() << ", "
                           << (operand.info().isDynamic() ? "Dynamic" : "Static") << ", "
                           << get_shape_str(operand.info().shape()) << std::endl;
  });
}

// TODO move this into Concat.cc
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

    if (input.info().isDynamic())
    {
      output.info().setDynamic();
      return;
    }

    input_shapes.emplace_back(input.shape());
  }

  ir::Shape out_shape = inferConcatShape(input_shapes, op.param());

  // re-sizing output shape
  output.info().shape(out_shape);
}

/*
 * DynamicInferer
  - Write methods except visit()
  - For visit() of each operator, find each op's C file
 */

void DynamicInferer::handleSimpleUnaryOp(const ir::Operation &op, const ir::OperandIndex input_ind)
{
  // check if output is not dynamic
  auto output_ind = op.getOutputs().at(0);
  auto *output = _tensor_registry->getITensor(output_ind);
  if (!output->is_dynamic())
    return;

  // getting output shape
  auto *input = _tensor_registry->getITensor(input_ind);
  auto output_shape = getShape(input);

  // set output shape and output buffer
  setShape(output, output_shape);

  // assert(output->buffer() == nullptr);
  _dynamic_tensor_manager->allocate(output_ind, output_shape);
  assert(output->buffer() != nullptr);
}

} // namespace shape_inference
} // namespace onert
