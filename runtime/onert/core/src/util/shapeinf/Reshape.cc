/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

bool isReshapableShape(const backend::ITensor *input, ir::Shape &shape)
{
  size_t input_elem_conut = 1;
  {
    for (size_t axis = 0; axis < input->num_dimensions(); axis++)
      input_elem_conut *= input->dimension(axis);
  }

  return (input_elem_conut == shape.num_elements());
}

} // namespace

namespace onert
{
namespace shape_inference
{

// StaticInferer at compilation time
void StaticInferer::visit(const ir::operation::Reshape &op)
{
  const auto input_idx{op.getInputs().at(ir::operation::Reshape::Input::INPUT)};
  const auto &input = _operands.at(input_idx);

  // get mutable output operand
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

  // if input is dynamic, output also becomes dynamic
  if (input.info().isDynamic())
  {
    output.info().setDynamic();
    return;
  }

  if (op.getInputs().size() == 1)
  {
    // no change on output shape
    return;
  }

  // Let's check the second input
  const auto shape_idx{op.getInputs().at(ir::operation::Reshape::Input::SHAPE)};
  const auto &shape = _operands.at(shape_idx);

  // if shape is from Const, TFLC put the shape of output into tensor
  if (shape.isConstant())
  {
    // no change on output shape
    return;
  }

  // if shape is NOT Const, set output shape to be dynamic_
  output.info().setDynamic();
}

// DynamicInferer at execution time
void DynamicInferer::visit(const ir::operation::Reshape &op)
{
  // check if output is not dynamic
  auto output_ind = op.getOutputs().at(0);
  auto *output = _tensor_registry->getITensor(output_ind);
  if (!output->is_dynamic())
    return;

  // from op, access the buffer of second input to read new shape
  auto new_shape_ind = op.getInputs().at(ir::operation::Reshape::Input::SHAPE);
  auto &new_shape_op = _operands.at(new_shape_ind);

  // if shape is from Const, TFLC put the shape of output into tensor
  if (new_shape_op.isConstant())
  {
    // no change on output shape
    return;
  }

  // getting output shape by reading new_shape tensor buffer
  auto new_shape = _tensor_registry->getITensor(new_shape_ind);
  assert(new_shape);

  int32_t *new_shape_buf = reinterpret_cast<int32_t *>(new_shape->buffer());
  assert(new_shape_buf);

  auto new_rank = new_shape->dimension(0);

  ir::Shape output_shape(new_rank);
  for (size_t d = 0; d < new_rank; d++)
    output_shape.dim(d) = new_shape_buf[d];

  // sanity check
  {
    auto input_ind = op.getInputs().at(ir::operation::Reshape::Input::INPUT);
    auto input = _tensor_registry->getITensor(input_ind);
    assert(input);

    if (!isReshapableShape(input, output_shape))
      throw std::runtime_error("Reshape: 2nd param is not compatible with the shape of input");
  }

  // set output shape and output buffer
  setShape(output, output_shape);

  _dynamic_tensor_manager->allocate(output_ind, output_shape);
  assert(output->buffer() != nullptr);
}

} // namespace shape_inference
} // namespace onert
