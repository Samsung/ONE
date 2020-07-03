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

void StaticInferer::visit(const ir::operation::BroadcastTo &op)
{
  const auto input_idx{op.getInputs().at(ir::operation::BroadcastTo::Input::INPUT)};
  const auto &input = _operands.at(input_idx);

  // get mutable output operand
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

  // if input is dynamic, output also becomes dynamic.
  if (input.info().isDynamic())
  {
    output.info().setDynamic();
    return;
  }

  const auto shape_idx{op.getInputs().at(ir::operation::BroadcastTo::Input::SHAPE)};
  const auto &shape = _operands.at(shape_idx);

  if (!shape.isConstant())
  {
    output.info().setDynamic();
    return;
  }

  // assert(shape.typeInfo().type() == ir::DataType::INT32);
  auto shape_buffer = reinterpret_cast<const int32_t *>(shape.data()->base());

  // re-sizing output shape
  ir::Shape new_shape = inferBroadcastToShape(shape.info().shape(), shape_buffer);
  output.info().shape(new_shape);
}

void DynamicInferer::visit(const ir::operation::BroadcastTo &op)
{
  auto output_ind = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_ind);

  auto input_idx = op.getInputs().at(ir::operation::BroadcastTo::INPUT);
  auto input = _tensor_registry->getITensor(input_idx);

  if ((!input->is_dynamic()) && (!output->is_dynamic()))
    return;

  auto shape_idx = op.getInputs().at(ir::operation::Tile::Input::MULTIPLES);
  const auto &shape = _tensor_registry->getITensor(shape_idx);

  assert(shape); // It shouldn't be 0.

  auto output_shape =
      inferBroadcastToShape(shape->getShape(), reinterpret_cast<const int32_t *>(shape->buffer()));

  // set output shape and output buffer
  _dynamic_tensor_manager->applyShape(output_ind, output_shape);
  assert(output->buffer() != nullptr);
}
} // namespace shape_inference
} // namespace onert
