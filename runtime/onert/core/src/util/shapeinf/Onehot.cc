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

ir::Shape onehotShape(const ir::Shape &input_shape, const int depth, int axis)
{
  assert(depth >= 0);
  const auto rank = input_shape.rank() + 1;
  ir::Shape newShape(rank);

  axis = (axis == -1) ? (rank - 1) : axis;

  for (int i = 0; i < rank; ++i)
  {
    if (i < axis)
    {
      newShape.dim(i) = input_shape.dim(i);
    }
    else if (i == axis)
    {
      newShape.dim(i) = depth;
    }
    else
    {
      newShape.dim(i) = input_shape.dim(i - 1);
    }
  }

  return newShape;
}

void StaticInferer::visit(const ir::operation::OneHot &op)
{
  const auto indice_idx{op.getInputs().at(ir::operation::OneHot::Input::INDICES)};
  const auto &indice = _operands.at(indice_idx);

  const auto depth = op.param().depth;
  const auto axis = op.param().axis;

  auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

  if (indice.info().isDynamic())
  {
    output.info().setDynamic();
    return;
  }

  // re-sizing output shape
  ir::Shape new_shape = onehotShape(indice.info().shape(), depth, axis);
  output.info().shape(new_shape);
}

void DynamicInferer::visit(const ir::operation::OneHot &op)
{
  auto output_ind = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_ind);

  auto indices_ind = op.getInputs().at(ir::operation::OneHot::INDICES);
  const auto &indices = _tensor_registry->getITensor(indices_ind);
  auto indices_shape = indices->getShape();

  if (!indices->is_dynamic())
  {
    return;
  }

  const auto depth = op.param().depth;
  const auto axis = op.param().axis;

  ir::Shape new_shape = onehotShape(indices_shape, depth, axis);
  _dynamic_tensor_manager->applyShape(output_ind, new_shape);
  assert(output->buffer() != nullptr);
}

} // namespace shape_inference
} // namespace onert
