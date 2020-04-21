/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

#include "kernels/Concatenation.h"

#include <cstring>
#include <stdexcept>

namespace luci_interpreter
{

namespace kernels
{

Concatenation::Concatenation(std::vector<const Tensor *> inputs, Tensor *output,
                             const ConcatenationParams &params)
    : _inputs(std::move(inputs)), _output(output), _params(params)
{
}

void Concatenation::configure()
{
  int32_t sum_axis = 0;
  for (const Tensor *input : _inputs)
  {
    sum_axis += input->shape().dim(_params.axis);
  }

  Shape output_shape = _inputs[0]->shape();
  output_shape.dim(_params.axis) = sum_axis;

  _output->resize(output_shape);
}

void Concatenation::execute() const
{
  switch (_inputs[0]->element_type())
  {
    case DataType::FLOAT32:
      evalGeneric<float>();
      break;
    case DataType::U8:
      throw std::runtime_error("Unsupported type.");
    case DataType::S8:
      evalGeneric<int8_t>();
      break;
    case DataType::S32:
      evalGeneric<int32_t>();
      break;
    case DataType::S64:
      evalGeneric<int64_t>();
      break;
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

template <typename T> void Concatenation::evalGeneric() const
{
  auto *output_data = _output->data<T>();

  const Shape &output_shape = _output->shape();

  int axis = _params.axis;
  if (axis < 0)
  {
    axis += output_shape.num_dims();
  }

  int32_t outer_size = 1;
  for (int i = 0; i < axis; ++i)
  {
    outer_size *= output_shape.dim(i);
  }

  int32_t inner_size = 1;
  for (int i = axis + 1; i < output_shape.num_dims(); ++i)
  {
    inner_size *= output_shape.dim(i);
  }

  T *output_ptr = output_data;
  for (int32_t i = 0; i < outer_size; ++i)
  {
    for (const Tensor *input : _inputs)
    {
      const int32_t slice_size = input->shape().dim(axis) * inner_size;
      const T *input_ptr = input->data<T>() + i * slice_size;
      std::memcpy(output_ptr, input_ptr, slice_size * sizeof(T));
      output_ptr += slice_size;
    }
  }
}

} // namespace kernels
} // namespace luci_interpreter
