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

#include "kernels/FullyConnected.h"

#include "kernels/Utils.h"

#include <stdexcept>

namespace luci_interpreter
{

namespace kernels
{

FullyConnected::FullyConnected(const Tensor *input, const Tensor *weights, const Tensor *bias,
                               Tensor *output, const FullyConnectedParams &params)
    : _input(input), _weights(weights), _bias(bias), _output(output), _params(params)
{
}

void FullyConnected::configure()
{
  const Shape &input_shape = _input->shape();
  const Shape &weights_shape = _weights->shape();

  const int32_t batch_size = input_shape.num_elements() / weights_shape.dim(1);
  const int32_t num_units = weights_shape.dim(0);

  _output->resize({batch_size, num_units});
}

void FullyConnected::execute() const
{
  switch (_weights->element_type())
  {
    case DataType::FLOAT32:
      if (_input->element_type() != DataType::FLOAT32)
      {
        throw std::runtime_error("Hybrid quantization is not supported.");
      }
      evalFloat();
      break;
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

// https://github.com/tensorflow/tensorflow/blob/v2.2.0-rc3/tensorflow/lite/kernels/internal/reference/fully_connected.h
void FullyConnected::evalFloat() const
{
  const auto *input_data = _input->data<float>();
  const auto *weights_data = _weights->data<float>();
  const auto *bias_data = _bias->data<float>();
  auto *output_data = _output->data<float>();

  const Shape &input_shape = _input->shape();
  const Shape &weights_shape = _weights->shape();

  float activation_min{};
  float activation_max{};
  calculateActivationRange(_params.activation, &activation_min, &activation_max);

  const int32_t accum_depth = weights_shape.dim(weights_shape.num_dims() - 1);
  const int32_t batches = input_shape.num_elements() / weights_shape.dim(1);
  const int32_t output_depth = weights_shape.dim(0);

  for (int32_t b = 0; b < batches; ++b)
  {
    for (int32_t out_c = 0; out_c < output_depth; ++out_c)
    {
      float sum = 0.0f;
      for (int32_t d = 0; d < accum_depth; ++d)
      {
        const float input_value = input_data[b * accum_depth + d];
        const float weights_value = weights_data[out_c * accum_depth + d];
        sum += input_value * weights_value;
      }
      sum += bias_data[out_c];
      output_data[output_depth * b + out_c] =
          activationFunctionWithMinMax(sum, activation_min, activation_max);
    }
  }
}

} // namespace kernels
} // namespace luci_interpreter
