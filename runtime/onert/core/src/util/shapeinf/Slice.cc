/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

#include <cmath>

namespace onert
{
namespace shape_inference
{

ir::Shape inferSliceShape(const ir::Shape &input_shape, const int32_t *begins, const int32_t *sizes,
                          uint32_t rank)
{
  ir::Shape out_shape(rank);

  for (uint32_t idx = 0; idx < rank; ++idx)
  {
    const auto input_dim = input_shape.dim(idx);

    // begin is zero-based
    auto begin = begins[idx];
    if (begin < 0)
    {
      begin += rank;
    }
    if (begin >= input_dim)
    {
      begin = input_dim - 1;
    }

    // size is one-based
    auto size = sizes[idx];
    assert(size >= -1);
    if (size == -1)
    {
      size = input_dim - begin;
    }
    // We are not sure what shape's dimension value is 0
    else if (size == 0)
    {
      size = 1;
    }

    // Clamping
    if (begin + size > input_dim)
    {
      size = input_dim - begin;
    }
    out_shape.dim(idx) = size;
  }

  return out_shape;
}

void StaticInferer::visit(const ir::operation::Slice &op)
{
  const auto input_index{op.getInputs().at(ir::operation::Slice::Input::INPUT)};
  const auto &input = _operands.at(input_index);
  const auto begins_index{op.getInputs().at(ir::operation::Slice::Input::BEGINS)};
  const auto &begins = _operands.at(begins_index);
  const auto sizes_index{op.getInputs().at(ir::operation::Slice::Input::SIZES)};
  const auto &sizes = _operands.at(sizes_index);
  const auto output_index = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_index);

  if (input.info().isDynamic() || begins.info().isDynamic() || sizes.info().isDynamic())
  {
    output.info().setDynamic();
    return;
  }

  // Whether input is constant or not does not affect whether output is dynamic or not
  if (!(begins.isConstant() && sizes.isConstant()))
  {
    output.info().setDynamic();
    return;
  }

  auto begins_buf = reinterpret_cast<const int32_t *>(begins.data()->base());
  auto sizes_buf = reinterpret_cast<const int32_t *>(sizes.data()->base());
  const auto rank = op.param().rank;

  ir::Shape new_shape = inferSliceShape(input.info().shape(), begins_buf, sizes_buf, rank);
  output.info().shape(new_shape);
}

void DynamicInferer::visit(const ir::operation::Slice &op)
{
  const auto input_index{op.getInputs().at(ir::operation::Slice::Input::INPUT)};
  const auto input = _tensor_registry->getITensor(input_index);
  const auto begins_index{op.getInputs().at(ir::operation::Slice::Input::BEGINS)};
  const auto begins = _tensor_registry->getITensor(begins_index);
  const auto sizes_index{op.getInputs().at(ir::operation::Slice::Input::SIZES)};
  const auto sizes = _tensor_registry->getITensor(sizes_index);
  auto output_index = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_index);

  if (!(input->is_dynamic() || begins->is_dynamic() || sizes->is_dynamic() || output->is_dynamic()))
  {
    return;
  }

  ir::Shape input_shape = getShape(input.get());
  auto begins_buf = reinterpret_cast<const int32_t *>(begins->buffer());
  auto sizes_buf = reinterpret_cast<const int32_t *>(sizes->buffer());
  const auto rank = input_shape.rank();

  ir::Shape new_shape =
      onert::shape_inference::inferSliceShape(input_shape, begins_buf, sizes_buf, rank);

  _dynamic_tensor_manager->applyShape(output_index, new_shape);
  assert(output->buffer() != nullptr);
}

} // namespace shape_inference
} // namespace onert
