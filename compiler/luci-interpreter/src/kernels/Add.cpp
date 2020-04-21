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

#include "kernels/Add.h"

#include "kernels/Utils.h"

#include <stdexcept>

namespace luci_interpreter
{

namespace kernels
{

Add::Add(const Tensor *input1, const Tensor *input2, Tensor *output, const AddParams &params)
    : _input1(input1), _input2(input2), _output(output), _params(params)
{
}

void Add::configure()
{
  const Shape &input1_shape = _input1->shape();
  const Shape &input2_shape = _input2->shape();

  const int num_input1_dims = input1_shape.num_dims();
  const int num_input2_dims = input2_shape.num_dims();
  const int num_out_dims = std::max(num_input1_dims, num_input2_dims);
  Shape output_shape(num_out_dims);

  _input1_strides.resize(num_out_dims);
  _input2_strides.resize(num_out_dims);
  _output_strides.resize(num_out_dims);

  int32_t input1_stride = 1;
  int32_t input2_stride = 1;
  int32_t output_stride = 1;

  for (int i = 0; i < num_out_dims; ++i)
  {
    const int32_t input1_dim = i < num_input1_dims ? input1_shape.dim(num_input1_dims - i - 1) : 1;
    const int32_t input2_dim = i < num_input2_dims ? input2_shape.dim(num_input2_dims - i - 1) : 1;
    assert(input1_dim == input2_dim || input1_dim == 1 || input2_dim == 1);
    const int32_t output_dim = std::max(input1_dim, input2_dim);
    output_shape.dim(num_out_dims - i - 1) = output_dim;

    _input1_strides[num_out_dims - i - 1] = input1_dim > 1 ? input1_stride : 0;
    _input2_strides[num_out_dims - i - 1] = input2_dim > 1 ? input2_stride : 0;
    _output_strides[num_out_dims - i - 1] = output_stride;

    input1_stride *= input1_dim;
    input2_stride *= input2_dim;
    output_stride *= output_dim;
  }

  _output->resize(output_shape);
}

void Add::execute() const
{
  switch (_input1->element_type())
  {
    case DataType::FLOAT32:
      evalFloat();
      break;
    case DataType::U8:
      evalQuantized();
      break;
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

template <typename T, typename Fn>
void Add::loop(int dim, const T *input1_data, const T *input2_data, T *output_data, Fn fn) const
{
  if (dim == _output->shape().num_dims())
  {
    *output_data = fn(*input1_data, *input2_data);
  }
  else
  {
    for (int i = 0; i < _output->shape().dim(dim); ++i)
    {
      loop(dim + 1, input1_data, input2_data, output_data, fn);
      input1_data += _input1_strides[dim];
      input2_data += _input2_strides[dim];
      output_data += _output_strides[dim];
    }
  }
}

void Add::evalFloat() const
{
  float activation_min{};
  float activation_max{};
  calculateActivationRange(_params.activation, &activation_min, &activation_max);

  const auto *input1_data = _input1->data<float>();
  const auto *input2_data = _input2->data<float>();
  auto *output_data = _output->data<float>();

  auto add = [=](float input1_val, float input2_val) {
    return activationFunctionWithMinMax(input1_val + input2_val, activation_min, activation_max);
  };

  loop(0, input1_data, input2_data, output_data, add);
}

// https://github.com/tensorflow/tensorflow/blob/v2.2.0-rc3/tensorflow/lite/kernels/internal/reference/add.h
void Add::evalQuantized() const
{
  const int32_t input1_offset = _input1->zero_point();
  const int32_t input2_offset = _input2->zero_point();
  const int32_t output_offset = _output->zero_point();

  const auto input1_scale = static_cast<double>(_input1->scale());
  const auto input2_scale = static_cast<double>(_input2->scale());
  const auto output_scale = static_cast<double>(_output->scale());

  const int32_t left_shift = 20;
  const double twice_max_input_scale = 2 * std::max(input1_scale, input2_scale);
  const double real_input1_multiplier = input1_scale / twice_max_input_scale;
  const double real_input2_multiplier = input2_scale / twice_max_input_scale;
  const double real_output_multiplier = twice_max_input_scale / ((1 << left_shift) * output_scale);

  int32_t input1_multiplier{}, input2_multiplier{}, output_multiplier{};
  int input1_shift{}, input2_shift{}, output_shift{};
  quantizeMultiplierSmallerThanOneExp(real_input1_multiplier, &input1_multiplier, &input1_shift);
  quantizeMultiplierSmallerThanOneExp(real_input2_multiplier, &input2_multiplier, &input2_shift);
  quantizeMultiplierSmallerThanOneExp(real_output_multiplier, &output_multiplier, &output_shift);

  int32_t activation_min{};
  int32_t activation_max{};
  calculateActivationRangeQuantized(_params.activation, _output, &activation_min, &activation_max);

  const auto *input1_data = _input1->data<uint8_t>();
  const auto *input2_data = _input2->data<uint8_t>();
  auto *output_data = _output->data<uint8_t>();

  auto quantized_add = [=](uint8_t input1_val, uint8_t input2_val) {
    const int32_t shifted_input1_val = (input1_val - input1_offset) * (1 << left_shift);
    const int32_t shifted_input2_val = (input2_val - input2_offset) * (1 << left_shift);

    const int32_t scaled_input1_val = multiplyByQuantizedMultiplierSmallerThanOneExp(
        shifted_input1_val, input1_multiplier, input1_shift);
    const int32_t scaled_input2_val = multiplyByQuantizedMultiplierSmallerThanOneExp(
        shifted_input2_val, input2_multiplier, input2_shift);

    const int32_t raw_sum = scaled_input1_val + scaled_input2_val;
    const int32_t raw_output =
        multiplyByQuantizedMultiplierSmallerThanOneExp(raw_sum, output_multiplier, output_shift) +
        output_offset;
    const int32_t clamped_output =
        activationFunctionWithMinMax(raw_output, activation_min, activation_max);
    return static_cast<uint8_t>(clamped_output);
  };

  loop(0, input1_data, input2_data, output_data, quantized_add);
}

} // namespace kernels
} // namespace luci_interpreter
