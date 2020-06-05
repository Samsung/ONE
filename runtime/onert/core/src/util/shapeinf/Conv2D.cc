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

#include "util/ShapeInference.h"

namespace onert
{
namespace shape_inference
{

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

void StaticInferer::visit(const ir::operation::Conv2D &op)
{
  const auto input_idx{op.getInputs().at(ir::operation::Conv2D::Input::INPUT)};
  const auto &input = _operands.at(input_idx);
  const auto ker_idx{op.getInputs().at(ir::operation::Conv2D::Input::KERNEL)};
  const auto &ker = _operands.at(ker_idx);
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

  if (input.info().isDynamic() || ker.info().isDynamic())
  {
    output.info().setDynamic();
    return;
  }

  // re-sizing output shape
  ir::Shape new_shape = inferConv2DShape(input.info().shape(), ker.info().shape(), op.param());
  output.info().shape(new_shape);
}

void DynamicInferer::visit(const ir::operation::Conv2D &op)
{
  // check if input is not dynamic
  auto input_ind = op.getInputs().at(ir::operation::Conv2D::INPUT);
  auto input = _tensor_registry->getITensor(input_ind);

  auto ker_ind = op.getInputs().at(ir::operation::Conv2D::KERNEL);
  auto ker = _tensor_registry->getITensor(ker_ind);

  if ((!input->is_dynamic()) && (!ker->is_dynamic()))
    return;

  ir::Shape input_shape = getShape(input.get());
  ir::Shape ker_shape = getShape(ker.get());

  auto output_ind = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_ind);

  ir::Shape output_shape = inferConv2DShape(input_shape, ker_shape, op.param());

  _dynamic_tensor_manager->applyShape(output_ind, output_shape);
  assert(output->buffer() != nullptr);
}

} // namespace shape_inference
} // namespace onert
