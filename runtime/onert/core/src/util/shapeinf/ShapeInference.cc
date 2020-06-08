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

#include <cassert>
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
ir::Shape inferAvgPoolShape(const ir::Shape &in_shape, const ir::operation::AvgPool2D::Param &param,
                            const ir::Layout layout)
{
  assert(layout == ir::Layout::NHWC);
  auto ifm_shape = in_shape.asFeature(layout);
  const auto out_h_w = calcConvLikeHeightAndWidth(ifm_shape.H, ifm_shape.W, param.kh, param.kw,
                                                  param.padding, param.stride);
  // Pooling don't change number of channels and batch size
  return ir::Shape{ifm_shape.N, out_h_w.first, out_h_w.second, ifm_shape.C};
}

// TODO move this when Conv2D.cc is created in util/shapeinf
ir::Shape inferConv2DShape(const ir::Shape &in_shape, const ir::Shape &ker_shape,
                           const ir::operation::Conv2D::Param &param, ir::Layout layout)
{
  assert(layout == ir::Layout::NHWC);
  auto ifm_shape = in_shape.asFeature(layout);

  // Kernel format is [depth_out, kernel_height, kernel_width, depth_in]
  auto kf_shape = ker_shape.asFeature(layout);
  assert(ifm_shape.C == kf_shape.C);

  const auto out_h_w = calcConvLikeHeightAndWidth(ifm_shape.H, ifm_shape.W, kf_shape.H, kf_shape.W,
                                                  param.padding, param.stride);

  return ir::Shape{ifm_shape.N, out_h_w.first, out_h_w.second, kf_shape.N};
}

// TODO move this when DepthwiseConv2D.cc is created in util/shapeinf
ir::Shape inferDepthwiseConv2DShape(const ir::Shape &in_shape, const ir::Shape &ker_shape,
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

  return ir::Shape{ifm_shape.N, out_h_w.first, out_h_w.second, kf_shape.C};
}

// TODO move this when MaxPool.cc is created in util/shapeinf
ir::Shape inferMaxPoolShape(const ir::Shape &in_shape, const ir::operation::MaxPool2D::Param &param,
                            const ir::Layout layout)
{
  assert(layout == ir::Layout::NHWC);
  auto ifm_shape = in_shape.asFeature(layout);
  const auto out_h_w = calcConvLikeHeightAndWidth(ifm_shape.H, ifm_shape.W, param.kh, param.kw,
                                                  param.padding, param.stride);
  // Pooling don't change number of channels and batch size
  return ir::Shape{ifm_shape.N, out_h_w.first, out_h_w.second, ifm_shape.C};
}

/*
  StaticInferer
  - Write methods except visit()
  - For visit() of each operator, find each op's C file
*/

void StaticInferer::handleBinaryArithmeticOp(const ir::Operation &op,
                                             const ir::OperandIndex lhs_idx,
                                             const ir::OperandIndex rhs_idx)
{
  const auto &lhs = _operands.at(lhs_idx);
  const auto &rhs = _operands.at(rhs_idx);

  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

  if (lhs.info().isDynamic() || rhs.info().isDynamic())
  {
    output.info().setDynamic();
    return;
  }

  // re-sizing output shape
  ir::Shape new_shape = inferEltwiseShape(lhs.info().shape(), rhs.info().shape());
  output.info().shape(new_shape);
}

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

  for (const auto &pair : _lowered_subgs)
  {
    const auto index = pair.first;
    const auto &lowered_subg = pair.second;
    VERBOSE(StaticInferer) << "SubGraph #" << index.value() << std::endl;
    lowered_subg->graph().operands().iterate(
        [&](const ir::OperandIndex &ind, const ir::Operand &operand) {
          VERBOSE(StaticInferer) << "Operand #" << ind.value() << ", "
                                 << (operand.info().isDynamic() ? "Dynamic" : "Static") << ", "
                                 << get_shape_str(operand.info().shape()) << std::endl;
        });
  }
}

/*
 * DynamicInferer
  - Write methods except visit()
  - For visit() of each operator, find each op's C file
 */

void DynamicInferer::handleBinaryArithmeticOp(const ir::Operation &op,
                                              const ir::OperandIndex lhs_idx,
                                              const ir::OperandIndex rhs_idx)
{
  auto lhs = _tensor_registry->getITensor(lhs_idx);
  auto lhs_shape = getShape(lhs.get());

  auto rhs = _tensor_registry->getITensor(rhs_idx);
  auto rhs_shape = getShape(rhs.get());

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

  ir::Shape new_shape = inferEltwiseShape(lhs_shape, rhs_shape);

  _dynamic_tensor_manager->applyShape(output_idx, new_shape);
  assert(output->buffer() != nullptr);
}

void DynamicInferer::handleSimpleUnaryOp(const ir::Operation &op, const ir::OperandIndex input_ind)
{
  // check if input is not dynamic
  auto input = _tensor_registry->getITensor(input_ind);
  auto output_shape = getShape(input.get());

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

} // namespace shape_inference
} // namespace onert
