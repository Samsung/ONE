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

void StaticInferer::visit(const ir::operation::OneHot &op)
{
  const auto indice_idx{op.getInputs().at(ir::operation::OneHot::Input::INDICES)};
  const auto &indice = _operands.at(indice_idx);
  const auto depth_idx{op.getInputs().at(ir::operation::OneHot::Input::DEPTH)};
  const auto &depth = _operands.at(depth_idx);

  const auto axis = op.param().axis;

  auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

  if (indice.info().isDynamic() || depth.info().isDynamic() || !depth.isConstant())
  {
    output.info().setDynamic();
    return;
  }

  const auto *depth_buf = reinterpret_cast<const int32_t *>(depth.data()->base());
  assert(depth_buf);
  // re-sizing output shape
  ir::Shape new_shape = onehotShape(indice.info().shape(), *depth_buf, axis);
  output.info().shape(new_shape);
}

void DynamicInferer::visit(const ir::operation::OneHot &op)
{
  auto output_ind = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_ind);

  auto indices_ind = op.getInputs().at(ir::operation::OneHot::INDICES);
  const auto &indices = _tensor_registry->getITensor(indices_ind);
  auto indices_shape = indices->getShape();

  auto depth_ind = op.getInputs().at(ir::operation::OneHot::DEPTH);
  const auto &depth = _tensor_registry->getITensor(depth_ind);

  if (!indices->is_dynamic() && !depth->is_dynamic())
  {
    return;
  }

  int32_t *depth_buf = reinterpret_cast<int32_t *>(depth->buffer());
  assert(depth_buf);
  const auto axis_val = op.param().axis;

  ir::Shape new_shape = onehotShape(indices_shape, *depth_buf, axis_val);
  _dynamic_tensor_manager->applyShape(output_ind, new_shape);
  assert(output->buffer() != nullptr);
}

} // namespace shape_inference
} // namespace onert
