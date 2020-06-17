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

ir::Shape convertShape(const int32_t *shape_buf, const int32_t rank,
                       const size_t total_num_elements)
{
  ir::Shape ret(rank);
  int32_t flatten_dim = ir::Shape::UNSPECIFIED_DIM;
  for (int32_t i = 0; i < rank; ++i)
  {
    if (shape_buf[i] < 0)
    {
      if (flatten_dim != ir::Shape::UNSPECIFIED_DIM)
        throw std::runtime_error("Reshape: 2nd param has special dim(for flatten) more than twice");
      flatten_dim = i;
      ret.dim(i) = 1;
    }
    else
    {
      ret.dim(i) = shape_buf[i];
    }
  }
  if (flatten_dim != ir::Shape::UNSPECIFIED_DIM)
    ret.dim(flatten_dim) = total_num_elements / ret.num_elements();

  // Check reshapable
  if (total_num_elements != static_cast<size_t>(ret.num_elements()))
    throw std::runtime_error("Reshape: 2nd param is not compatible with the shape of input");

  return ret;
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

  ir::Shape new_shape;
  if (shape.isConstant())
  {
    const auto *shape_buf = reinterpret_cast<const int32_t *>(shape.data()->base());
    assert(shape_buf);

    new_shape = convertShape(shape_buf, shape.shape().num_elements(), input.shape().num_elements());

    // if shape is from Const, TFLC put the shape of output into tensor
    if (new_shape != output.shape())
    {
      // change on output shape
      output.info().shape(new_shape);
    }
  }
  else
  {
    // if shape is NOT Const, set output shape to be dynamic_
    output.info().setDynamic();
  }
}

// DynamicInferer at execution time
void DynamicInferer::visit(const ir::operation::Reshape &op)
{
  // check if output is not dynamic
  auto output_ind = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_ind);

  auto input_ind = op.getInputs().at(ir::operation::Reshape::Input::INPUT);
  auto input = _tensor_registry->getITensor(input_ind);

  /*
    Here, the state after compilation (satic shape inference) could be one of the following:

              input1     input2      output     execution-time shape inf required
              -----------------------------     --------------------------------
      case 1) static     const       static       X
      case 2) static    placeholder  dynamic      O
      case 3) dynamic    const       dynamic      O
      case 4) dynamic   placeholder  dynamic      O

    Then nnfw_apply_tensorinf() could change input dynamic.
    So, in this method, we could have one more state and we have to re-calculate shape
    for this shape.

      case 5) dynamic    const       static       O

    So, only when both input1 and ouput are static, we can skip dynamic shape inference.
  */
  if ((!input->is_dynamic()) && (!output->is_dynamic()))
    return;

  // from op, access the buffer of second input to read new shape
  auto new_shape_ind = op.getInputs().at(ir::operation::Reshape::Input::SHAPE);

  // getting output shape by reading new_shape tensor buffer
  auto new_shape = _tensor_registry->getITensor(new_shape_ind);
  assert(new_shape);

  int32_t *new_shape_buf = reinterpret_cast<int32_t *>(new_shape->buffer());
  assert(new_shape_buf);

  auto output_shape = convertShape(new_shape_buf, new_shape->getShape().num_elements(),
                                   input->getShape().num_elements());

  // if shape is changed, change output shape and reallocate output tensor memory
  if (output_shape != output->getShape() || output->buffer() == nullptr)
  {
    // change on output shape
    _dynamic_tensor_manager->applyShape(output_ind, output_shape);
  }
  assert(output->buffer() != nullptr);
}

} // namespace shape_inference
} // namespace onert
