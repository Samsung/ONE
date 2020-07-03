/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

#include <cmath>

namespace onert
{
namespace shape_inference
{

void StaticInferer::visit(const ir::operation::StridedSlice &op)
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

  if (input.info().isDynamic() || starts.info().isDynamic() || ends.info().isDynamic() ||
      strides.info().isDynamic())
  {
    output.info().setDynamic();
    return;
  }

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

  auto op_params = buildStridedSliceParams(starts_buf, ends_buf, strides_buf, begin_mask, end_mask,
                                           shrink_axis_mask, rank);

  ir::Shape new_shape = inferStridedSliceShape(input.info().shape(), op_params, rank);
  output.info().shape(new_shape);
}

void DynamicInferer::visit(const ir::operation::StridedSlice &op)
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

  auto op_params = buildStridedSliceParams(reinterpret_cast<uint32_t *>(starts->buffer()),
                                           reinterpret_cast<uint32_t *>(ends->buffer()),
                                           reinterpret_cast<uint32_t *>(strides->buffer()),
                                           begin_mask, end_mask, shrink_axis_mask, rank);

  auto output_index = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_index);

  ir::Shape output_shape =
      onert::shape_inference::inferStridedSliceShape(input_shape, op_params, rank);

  _dynamic_tensor_manager->applyShape(output_index, output_shape);
  assert(output->buffer() != nullptr);
}

} // namespace shape_inference
} // namespace onert
