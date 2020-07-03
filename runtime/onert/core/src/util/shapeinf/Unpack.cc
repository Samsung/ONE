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

void StaticInferer::visit(const ir::operation::Unpack &op)
{
  const auto input_idx{op.getInputs().at(0)};
  const auto &input = _operands.at(input_idx);
  const auto num = op.param().num;

  // if input is dynamic, output also becomes dynamic
  if (input.info().isDynamic())
  {
    for (int out_tensor_idx = 0; out_tensor_idx < num; out_tensor_idx++)
    {
      const auto output_idx = op.getOutputs().at(out_tensor_idx);
      ir::Operand &output = _operands.at(output_idx);
      output.info().setDynamic();
    }

    return;
  }

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

    return;
  }

  ir::Shape new_shape = unpackShapes(input.info().shape(), axis, rank);

  // re-sizing output shape
  for (int out_tensor_idx = 0; out_tensor_idx < num; out_tensor_idx++)
  {
    const auto output_idx = op.getOutputs().at(out_tensor_idx);
    ir::Operand &output = _operands.at(output_idx);
    output.info().shape(new_shape);
  }
}

void DynamicInferer::visit(const ir::operation::Unpack &op)
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

  ir::Shape new_shape = unpackShapes(input_shape, axis, rank);

  for (int out_tensor_idx = 0; out_tensor_idx < num; out_tensor_idx++)
  {
    auto output_ind = op.getOutputs().at(out_tensor_idx);
    auto output = _tensor_registry->getITensor(output_ind);

    _dynamic_tensor_manager->applyShape(output_ind, new_shape);

    assert(output->buffer() != nullptr);
  }
}

} // namespace shape_inference
} // namespace onert
