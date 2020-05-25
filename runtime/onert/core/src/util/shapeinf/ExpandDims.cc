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

ir::Shape inferExpandDimsShape(const ir::Shape &in_shape, int32_t axis)
{
  ir::Shape out_shape(in_shape.rank() + 1);

  axis = ((axis >= 0) ? axis : /* when axis < 0 */ (out_shape.rank() + axis));
  if (!(0 <= axis && axis <= in_shape.rank()))
    throw std::runtime_error("axis of dim is out of range");

  for (int x = 0, out_x = 0; out_x < out_shape.rank(); ++out_x)
  {
    if (out_x == axis)
      out_shape.dim(out_x) = 1;
    else
      out_shape.dim(out_x) = in_shape.dim(x++);
  }

  return out_shape;
}

void StaticInferer::visit(const ir::operation::ExpandDims &op)
{
  const auto input_idx{op.getInputs().at(ir::operation::ExpandDims::Input::INPUT)};
  const auto &input = _operands.at(input_idx);
  const auto axis_idx{op.getInputs().at(ir::operation::ExpandDims::Input::AXIS)};
  const auto &axis = _operands.at(axis_idx);
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

  if (input.info().isDynamic())
  {
    output.info().setDynamic();
    return;
  }

  if (!axis.isConstant())
  {
    output.info().setDynamic();
    return;
  }

  // even when axis is constant, output shape should be recalculated since user might call
  // nnfw_apply_tensorinfo(input, some_new_shape)
  auto axis_buf = reinterpret_cast<const int32_t *>(axis.data()->base());
  assert(axis_buf);

  // re-sizing output shape
  ir::Shape new_shape = inferExpandDimsShape(input.info().shape(), axis_buf[0]);
  output.info().shape(new_shape);
}

void DynamicInferer::visit(const ir::operation::ExpandDims &op)
{
  // check if output is not dynamic
  auto output_ind = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_ind).get();
  if (!output->is_dynamic())
    return;

  // getting output shape
  auto input_ind = op.getInputs().at(ir::operation::ExpandDims::INPUT);
  auto input = _tensor_registry->getITensor(input_ind).get();
  ir::Shape input_shape = getShape(input);

  auto axis_ind = op.getInputs().at(ir::operation::ExpandDims::AXIS);
  auto axis = _tensor_registry->getITensor(axis_ind);
  auto axis_buf = reinterpret_cast<const int32_t *>(axis->buffer());
  assert(axis_buf);

  auto output_shape = onert::shape_inference::inferExpandDimsShape(input_shape, axis_buf[0]);

  // set output shape and output buffer
  setShape(output, output_shape);

  // assert(output->buffer() == nullptr);
  _dynamic_tensor_manager->allocate(output_ind, output_shape);
  assert(output->buffer() != nullptr);
}

} // namespace shape_inference
} // namespace onert
