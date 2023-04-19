/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kernels/Reshape.h"

#include "kernels/Utils.h"

#include <cassert>
#include <cstring>

namespace luci_interpreter
{

namespace kernels
{

static Shape extractShapeFromTensor(const Tensor *tensor)
{
  Shape shape(tensor->shape().num_elements());
  if (tensor->element_type() == DataType::S32)
  {
    const auto *shape_data = tensor->data<int32_t>();
    for (int i = 0; i < tensor->shape().num_elements(); ++i)
    {
      shape.dim(i) = shape_data[i];
    }
  }
  else if (tensor->element_type() == DataType::S64)
  {
    const auto *shape_data = tensor->data<int64_t>();
    for (int i = 0; i < tensor->shape().num_elements(); ++i)
    {
      shape.dim(i) = static_cast<int32_t>(shape_data[i]);
    }
  }
  else
  {
    LUCI_INTERPRETER_CHECK(false);
  }
  return shape;
}

static void resolveUnknownDimension(const Shape &input_shape, Shape *output_shape)
{
  const int32_t num_input_elements = input_shape.num_elements();
  int32_t num_output_elements = 1;
  int unknown_dim_index = -1;
  for (int i = 0; i < output_shape->num_dims(); ++i)
  {
    const int32_t value = output_shape->dim(i);
    if (value == -1)
    {
      assert(unknown_dim_index == -1);
      unknown_dim_index = i;
    }
    else
    {
      num_output_elements *= value;
    }
  }
  if (unknown_dim_index != -1)
  {
    output_shape->dim(unknown_dim_index) = num_input_elements / num_output_elements;
    num_output_elements *= output_shape->dim(unknown_dim_index);
  }
  assert(num_output_elements == num_input_elements);
}

Reshape::Reshape(const Tensor *input, const Tensor *shape, Tensor *output)
  : Kernel({input, shape}, {output})
{
}

void Reshape::configure()
{
  Shape output_shape = extractShapeFromTensor(shape());
  resolveUnknownDimension(input()->shape(), &output_shape);
  output()->resize(output_shape);
}

void Reshape::execute() const
{
  const auto *input_data = input()->data<void>();
  auto *output_data = output()->data<void>();

  const size_t element_size = getDataTypeSize(input()->element_type());
  const int32_t num_elements = input()->shape().num_elements();
  std::memcpy(output_data, input_data, num_elements * element_size);
}

} // namespace kernels
} // namespace luci_interpreter
