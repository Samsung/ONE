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

void StaticInferer::visit(const ir::operation::Tile &op)
{
  const auto input_idx{op.getInputs().at(ir::operation::Tile::Input::INPUT)};
  const auto &input = _operands.at(input_idx);

  const auto multiplier_idx{op.getInputs().at(ir::operation::Tile::Input::MULTIPLES)};
  const auto &multiplier = _operands.at(multiplier_idx);

  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

  if (input.info().isDynamic())
  {
    output.info().setDynamic();
    return;
  }

  if (!multiplier.isConstant())
  {
    output.info().setDynamic();
    return;
  }

  auto multiplier_buffer = reinterpret_cast<const int32_t *>(multiplier.data()->base());
  assert(multiplier_buffer);

  // re-sizing output shape
  auto new_shape = inferTileShape(input.info().shape(), multiplier_buffer);
  output.info().shape(new_shape);
}

void DynamicInferer::visit(const ir::operation::Tile &op)
{
  auto output_ind = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_ind);

  auto input_idx = op.getInputs().at(ir::operation::Tile::Input::INPUT);
  auto input = _tensor_registry->getITensor(input_idx);

  auto multiplier_idx = op.getInputs().at(ir::operation::Tile::Input::MULTIPLES);
  auto multiplier = _tensor_registry->getITensor(multiplier_idx);

  if ((!input->is_dynamic()) && (!output->is_dynamic()))
    return;

  auto input_shape = input->getShape();
  auto multiplier_buffer = reinterpret_cast<const int32_t *>(multiplier->buffer());
  assert(multiplier_buffer);

  auto output_shape = inferTileShape(input_shape, multiplier_buffer);

  // set output shape and output buffer
  _dynamic_tensor_manager->applyShape(output_ind, output_shape);
  assert(output->buffer() != nullptr);
}
}
}
