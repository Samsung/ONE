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

void StaticInferer::visit(const ir::operation::FullyConnected &op)
{
  const auto input_idx{op.getInputs().at(ir::operation::FullyConnected::Input::INPUT)};
  const auto &input = _operands.at(input_idx);

  const auto ker_idx{op.getInputs().at(ir::operation::FullyConnected::Input::WEIGHT)};
  const auto &ker = _operands.at(ker_idx);

  // get mutable output operand
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

  // if input or ker is dynamic, output also becomes dynamic
  if (input.info().isDynamic() || ker.info().isDynamic())
  {
    output.info().setDynamic();
    return;
  }

  // re-sizing output shape
  ir::Shape new_shape = inferFullyConnectedShape(input.info().shape(), ker.info().shape());
  output.info().shape(new_shape);
}

void DynamicInferer::visit(const ir::operation::FullyConnected &op)
{
  const auto input_idx{op.getInputs().at(ir::operation::FullyConnected::Input::INPUT)};
  const auto &input = _tensor_registry->getITensor(input_idx);

  const auto ker_idx{op.getInputs().at(ir::operation::FullyConnected::Input::WEIGHT)};
  const auto &ker = _tensor_registry->getITensor(ker_idx);

  if (!input->is_dynamic() && !ker->is_dynamic())
    return;

  auto input_shape = input->getShape();
  auto ker_shape = ker->getShape();

  ir::Shape new_shape = inferFullyConnectedShape(input_shape, ker_shape);

  auto output_ind = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_ind);

  _dynamic_tensor_manager->applyShape(output_ind, new_shape);
  assert(output->buffer() != nullptr);
}

} // namespace shape_inference
} // namespace onert
