
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

void StaticInferer::visit(const ir::operation::Pack &op)
{
  bool is_any_of_inputs_dynamic = [&]() -> bool {
    for (uint32_t i = 0; i < op.getInputs().size(); ++i)
    {
      const auto &input = _operands.at(op.getInputs().at(i));
      if (input.info().isDynamic())
      {
        return true;
      }
    }
    return false;
  }();

  const auto input_idx{op.getInputs().at(0)};
  const auto &input = _operands.at(input_idx);

  // get mutable output operand
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

  // if input is dynamic, output also becomes dynamic
  if (is_any_of_inputs_dynamic)
  {
    output.info().setDynamic();
    return;
  }

  const auto rank = input.shape().rank() + 1;
  const auto axis = ((op.param().axis < 0) ? rank + op.param().axis : op.param().axis);
  const auto num = op.param().num;

  assert(0 <= axis && axis < rank);

  // re-sizing output shape
  ir::Shape new_shape = packShapes(input.info().shape(), axis, rank, num);
  output.info().shape(new_shape);
}

void DynamicInferer::visit(const ir::operation::Pack &op)
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

  ir::Shape new_shape = packShapes(input_shape, axis, rank, num);

  _dynamic_tensor_manager->applyShape(output_ind, new_shape);
  assert(output->buffer() != nullptr);
}

} // namespace shape_inference
} // namespace onert
