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

#include <tensorflow/lite/kernels/internal/reference/add.h>
#include <tensorflow/lite/kernels/internal/reference/process_broadcast_shapes.h>

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
  _output->resize(calculateShapeForBroadcast(_input1->shape(), _input2->shape()));
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

void Add::evalFloat() const
{
  float activation_min{};
  float activation_max{};
  calculateActivationRange(_params.activation, &activation_min, &activation_max);

  tflite::ArithmeticParams params{};
  params.float_activation_min = activation_min;
  params.float_activation_max = activation_max;

  const bool need_broadcast = tflite::reference_ops::ProcessBroadcastShapes(
      convertShape(_input1->shape()), convertShape(_input2->shape()), &params);

  if (need_broadcast)
  {
    tflite::reference_ops::BroadcastAdd4DSlow(
        params, convertShape(_input1->shape()), _input1->data<float>(),
        convertShape(_input2->shape()), _input2->data<float>(), convertShape(_output->shape()),
        _output->data<float>());
  }
  else
  {
    tflite::reference_ops::Add(params, convertShape(_input1->shape()), _input1->data<float>(),
                               convertShape(_input2->shape()), _input2->data<float>(),
                               convertShape(_output->shape()), _output->data<float>());
  }
}

void Add::evalQuantized() const
{
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

  tflite::ArithmeticParams params{};
  params.left_shift = left_shift;
  params.input1_offset = -_input1->zero_point(); // Note the '-'.
  params.input1_multiplier = input1_multiplier;
  params.input1_shift = input1_shift;
  params.input2_offset = -_input2->zero_point(); // Note the '-'.
  params.input2_multiplier = input2_multiplier;
  params.input2_shift = input2_shift;
  params.output_offset = _output->zero_point();
  params.output_multiplier = output_multiplier;
  params.output_shift = output_shift;
  params.quantized_activation_min = activation_min;
  params.quantized_activation_max = activation_max;

  const bool need_broadcast = tflite::reference_ops::ProcessBroadcastShapes(
      convertShape(_input1->shape()), convertShape(_input2->shape()), &params);

  if (need_broadcast)
  {
    tflite::reference_ops::BroadcastAdd4DSlow(
        params, convertShape(_input1->shape()), _input1->data<uint8_t>(),
        convertShape(_input2->shape()), _input2->data<uint8_t>(), convertShape(_output->shape()),
        _output->data<uint8_t>());
  }
  else
  {
    tflite::reference_ops::Add(params, convertShape(_input1->shape()), _input1->data<uint8_t>(),
                               convertShape(_input2->shape()), _input2->data<uint8_t>(),
                               convertShape(_output->shape()), _output->data<uint8_t>());
  }
}

} // namespace kernels
} // namespace luci_interpreter
