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

#include "kernels/Softmax.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace luci_interpreter
{

namespace kernels
{

Softmax::Softmax(const Tensor *input, Tensor *output, const SoftmaxParams &params)
    : _input(input), _output(output), _params(params)
{
}

void Softmax::configure() { _output->resize(_input->shape()); }

void Softmax::execute() const
{
  switch (_input->element_type())
  {
    case DataType::FLOAT32:
      evalFloat();
      break;
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

// https://github.com/tensorflow/tensorflow/blob/v2.2.0-rc3/tensorflow/lite/kernels/internal/reference/softmax.h
void Softmax::evalFloat() const
{
  const auto *input_data = _input->data<float>();
  auto *output_data = _output->data<float>();

  const Shape &input_shape = _input->shape();

  const int32_t trailing_dim = input_shape.num_dims() - 1;
  const int32_t depth = input_shape.dim(trailing_dim);
  const int32_t outer_size = input_shape.num_elements() / depth;

  for (int32_t i = 0; i < outer_size; ++i)
  {
    float max_val = std::numeric_limits<float>::lowest();
    for (int32_t c = 0; c < depth; ++c)
    {
      const float val = input_data[i * depth + c];
      if (val > max_val)
      {
        max_val = val;
      }
    }

    float sum = 0.0f;
    for (int32_t c = 0; c < depth; ++c)
    {
      const float val = input_data[i * depth + c];
      sum += std::exp((val - max_val) * _params.beta);
    }

    for (int32_t c = 0; c < depth; ++c)
    {
      const float val = input_data[i * depth + c];
      output_data[i * depth + c] = std::exp((val - max_val) * _params.beta) / sum;
    }
  }
}

} // namespace kernels
} // namespace luci_interpreter
