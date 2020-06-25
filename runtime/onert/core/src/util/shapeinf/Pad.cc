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

// helper function
namespace
{

using namespace onert;

ir::Shape inferPadShape(const ir::Shape &in_shape, const int32_t *pad_buf, const size_t num_pads)
{
  assert(num_pads % 2 == 0);
  const int32_t rank = num_pads / 2;

  ir::Shape ret(rank);
  for (int32_t i = 0; i < rank; ++i)
  {
    const auto before_padding = pad_buf[i * 2];
    const auto after_padding = pad_buf[i * 2 + 1];

    ret.dim(i) = in_shape.dim(i) + before_padding + after_padding;
  }

  return ret;
}

} // namespace

namespace onert
{
namespace shape_inference
{

// StaticInferer at compilation time
void StaticInferer::visit(const ir::operation::Pad &op)
{
  const auto input_idx{op.getInputs().at(ir::operation::Pad::Input::INPUT)};
  const auto &input = _operands.at(input_idx);

  const auto pad_idx{op.getInputs().at(ir::operation::Pad::Input::PAD)};
  const auto &pad = _operands.at(pad_idx);

  // get mutable output operand
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

  // if input is dynamic or pad is dynamic, output also becomes dynamic
  if (input.info().isDynamic() || pad.info().isDynamic())
  {
    output.info().setDynamic();
    return;
  }

  // if pad is not constant, output also becomes dynamic
  if (!pad.isConstant())
  {
    output.info().setDynamic();
    return;
  }

  // re-sizing output shape
  const auto new_shape =
      inferPadShape(input.shape(), reinterpret_cast<const int32_t *>(pad.data()->base()),
                    pad.shape().num_elements());
  output.info().shape(new_shape);
}

// DynamicInferer at execution time
void DynamicInferer::visit(const ir::operation::Pad &op)
{
  // check if output is not dynamic
  auto output_ind = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_ind);

  auto input_ind = op.getInputs().at(ir::operation::Pad::Input::INPUT);
  auto input = _tensor_registry->getITensor(input_ind);

  auto pad_ind = op.getInputs().at(ir::operation::Pad::Input::PAD);
  auto pad = _tensor_registry->getITensor(pad_ind);

  // check if input and output are not dynamic
  if ((!input->is_dynamic()) && (!output->is_dynamic()))
    return;

  int32_t *pad_buf = reinterpret_cast<int32_t *>(pad->buffer());
  assert(pad_buf);

  auto output_shape = inferPadShape(input->getShape(), pad_buf, pad->getShape().num_elements());

  // change output shape and reallocate output tensor memory
  _dynamic_tensor_manager->applyShape(output_ind, output_shape);
  assert(output->buffer() != nullptr);
}

} // namespace shape_inference
} // namespace onert
