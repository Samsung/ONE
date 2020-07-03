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
#include "ir/Data.h"

namespace onert
{
namespace shape_inference
{

void StaticInferer::visit(const ir::operation::Fill &op)
{
  const auto input_idx{op.getInputs().at(ir::operation::Fill::Input::INPUT)};
  const auto &input = _operands.at(input_idx);
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

  if (input.info().isDynamic())
  {
    output.info().setDynamic();
    return;
  }

  if (!input.isConstant())
  {
    output.info().setDynamic();
    return;
  }

  assert(input.typeInfo().type() == ir::DataType::INT32);

  auto input_buf = reinterpret_cast<const int32_t *>(input.data()->base());
  assert(input_buf);

  // re-sizing output shape
  ir::Shape new_shape = inferFillShape(input.info().shape(), input_buf);
  output.info().shape(new_shape);
}

void DynamicInferer::visit(const ir::operation::Fill &op)
{
  // check if output is not dynamic
  auto output_ind = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_ind);
  auto input_ind = op.getInputs().at(ir::operation::Fill::Input::INPUT);
  auto input = _tensor_registry->getITensor(input_ind);
  ir::Shape input_shape = input->getShape();

  if ((!input->is_dynamic()) && (!output->is_dynamic()))
    return;

  assert(input.get()->data_type() == ir::DataType::INT32);

  auto input_buf = reinterpret_cast<const int32_t *>(input->buffer());
  assert(input_buf);

  auto output_shape = onert::shape_inference::inferFillShape(input_shape, input_buf);

  _dynamic_tensor_manager->applyShape(output_ind, output_shape);
  assert(output->buffer() != nullptr);
}

} // namespace shape_inference
} // namespace onert
