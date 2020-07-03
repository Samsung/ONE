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
  // nnfw_set_input_tensorinfo(input, some_new_shape)
  auto axis_buf = reinterpret_cast<const int32_t *>(axis.data()->base());
  assert(axis_buf);

  // re-sizing output shape
  ir::Shape new_shape = inferExpandDimsShape(input.info().shape(), axis_buf[0]);
  output.info().shape(new_shape);
}

void DynamicInferer::visit(const ir::operation::ExpandDims &op)
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

  auto output_shape = onert::shape_inference::inferExpandDimsShape(input_shape, axis_buf[0]);

  _dynamic_tensor_manager->applyShape(output_ind, output_shape);
  assert(output->buffer() != nullptr);
}

} // namespace shape_inference
} // namespace onert
