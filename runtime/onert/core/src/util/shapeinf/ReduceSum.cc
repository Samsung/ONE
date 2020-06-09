/* Copyright (c) 2017 The TensorFlow Authors. All Rights Reserved.
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

ir::Shape inferReduceShapes(const ir::Shape &input_shape, const std::vector<int> &axes,
                            bool keep_dims)
{
  int num_axis = axes.size();
  int input_num_dims = input_shape.rank();
  if (input_num_dims == 0)
  {
    ir::Shape out_shape(0);
    return out_shape;
  }
  if (keep_dims)
  {
    ir::Shape out_shape;
    for (int idx = 0; idx < input_num_dims; ++idx)
    {
      bool is_axis = false;
      for (int axis_idx = 0; axis_idx < num_axis; ++axis_idx)
      {
        if (axes[axis_idx] == idx || axes[axis_idx] + input_num_dims == idx)
        {
          is_axis = true;
          break;
        }
      }
      if (is_axis)
      {
        out_shape.append(1);
      }
      else
      {
        out_shape.append(input_shape.dim(idx));
      }
    }
    return out_shape;
  }
  else
  {
    // Calculates size of reducing axis.
    int num_reduce_axis = num_axis;
    for (int i = 0; i < num_axis; ++i)
    {
      int current = axes[i];
      if (current < 0)
      {
        current += input_num_dims;
      }
      assert(0 <= current && current < input_num_dims);
      for (int j = 0; j < i; ++j)
      {
        int previous = axes[j];
        if (previous < 0)
        {
          previous += input_num_dims;
        }
        if (current == previous)
        {
          --num_reduce_axis;
          break;
        }
      }
    }
    // Determines output dimensions.
    ir::Shape out_shape;
    int num_skip_axis = 0;
    for (int idx = 0; idx < input_num_dims; ++idx)
    {
      bool is_axis = false;
      for (int axis_idx = 0; axis_idx < num_axis; ++axis_idx)
      {
        if (axes[axis_idx] == idx || axes[axis_idx] + input_num_dims == idx)
        {
          ++num_skip_axis;
          is_axis = true;
          break;
        }
      }
      if (!is_axis)
      {
        out_shape.append(input_shape.dim(idx));
      }
    }
    return out_shape;
  }
}

void StaticInferer::visit(const ir::operation::ReduceSum &op)
{
  const auto input_idx{op.getInputs().at(0)};
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

  const auto axes = op.param().axes;
  const auto keep_dims = op.param().keep_dims;

  // re-sizing output shape
  ir::Shape new_shape = inferReduceShapes(input.info().shape(), axes, keep_dims);
  output.info().shape(new_shape);
}

void DynamicInferer::visit(const ir::operation::ReduceSum &op)
{
  const auto input_idx{op.getInputs().at(0)};
  const auto &input = _tensor_registry->getITensor(input_idx);
  auto input_shape = getShape(input.get());

  if (!input->is_dynamic())
    return;

  const auto axes = op.param().axes;
  const auto keep_dims = op.param().keep_dims;

  auto output_ind = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_ind);

  ir::Shape new_shape = inferReduceShapes(input_shape, axes, keep_dims);

  _dynamic_tensor_manager->applyShape(output_ind, new_shape);
  assert(output->buffer() != nullptr);
}

} // namespace shape_inference
} // namespace onert
