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

void StaticInferer::visit(const ir::operation::Gather &op)
{
  const auto input_idx{op.getInputs().at(ir::operation::Gather::Input::INPUT)};
  const auto &input = _operands.at(input_idx);

  // get mutable output operand
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

  const auto indices_idx{op.getInputs().at(ir::operation::Gather::Input::INDICES)};
  const auto &indices = _operands.at(indices_idx);

  // if input is dynamic, output also becomes dynamic
  if (input.info().isDynamic() || indices.info().isDynamic())
  {
    output.info().setDynamic();
    return;
  }

  const auto rank = input.info().shape().rank();
  const auto axis = ((op.param().axis < 0) ? rank + op.param().axis : op.param().axis);

  assert(0 <= axis && axis < rank);

  // re-sizing output shape
  ir::Shape new_shape = gatherShapes(input.info().shape(), indices.info().shape(), axis, rank);
  output.info().shape(new_shape);
}

void DynamicInferer::visit(const ir::operation::Gather &op)
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

  ir::Shape new_shape = gatherShapes(input_shape, indices_shape, axis, rank);

  auto output_ind = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_ind);

  _dynamic_tensor_manager->applyShape(output_ind, new_shape);
  assert(output->buffer() != nullptr);
}

} // namespace shape_inference
} // namespace onert
