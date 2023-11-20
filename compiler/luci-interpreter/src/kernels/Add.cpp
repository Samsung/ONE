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

#include "kernels/BinaryOpCommon.h"
#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/reference/add.h>
#include <tensorflow/lite/kernels/internal/reference/process_broadcast_shapes.h>

#include <stdexcept>

namespace luci_interpreter
{
namespace kernels
{

Add::Add(const Tensor *input1, const Tensor *input2, Tensor *output, const AddParams &params)
  : KernelWithParams<AddParams>({input1, input2}, {output}, params)
{
}

void Add::configure()
{
  LUCI_INTERPRETER_CHECK(input1()->element_type() == input2()->element_type());
  LUCI_INTERPRETER_CHECK(input1()->element_type() == output()->element_type());
  if (input1()->element_type() == DataType::S16)
  {
    LUCI_INTERPRETER_CHECK(input1()->zero_points().size() == 1 &&
                           input2()->zero_points().size() == 1);
    LUCI_INTERPRETER_CHECK(input1()->zero_point() == 0 && input2()->zero_point() == 0 &&
                           output()->zero_point() == 0);
  }

  output()->resize(calculateShapeForBroadcast(input1()->shape(), input2()->shape()));
}

void Add::execute() const
{
  switch (input1()->element_type())
  {
    case DataType::FLOAT32:
      evalFloat();
      break;
    case DataType::S64:
      evalInteger<int64_t>();
      break;
    case DataType::S32:
      evalInteger<int32_t>();
      break;
    case DataType::U8:
      evalQuantized();
      break;
    case DataType::S16:
      evalQuantizedS16();
      break;
    default:
      throw std::runtime_error("luci-intp Add Unsupported type.");
  }
}

void Add::evalFloat() const
{
  tflite::ArithmeticParams params{};
  fillArithmeticActivationRange<float>(params, _params.activation);

  const bool need_broadcast = tflite::reference_ops::ProcessBroadcastShapes(
    getTensorShape(input1()), getTensorShape(input2()), &params);

  if (need_broadcast)
  {
    tflite::reference_ops::BroadcastAdd4DSlow(
      params, getTensorShape(input1()), getTensorData<float>(input1()), getTensorShape(input2()),
      getTensorData<float>(input2()), getTensorShape(output()), getTensorData<float>(output()));
  }
  else
  {
    tflite::reference_ops::Add(params, getTensorShape(input1()), getTensorData<float>(input1()),
                               getTensorShape(input2()), getTensorData<float>(input2()),
                               getTensorShape(output()), getTensorData<float>(output()));
  }
}

template <typename T> void Add::evalInteger() const
{
  tflite::ArithmeticParams params{};
  fillArithmeticActivationRange<T>(params, _params.activation);

  const bool need_broadcast = tflite::reference_ops::ProcessBroadcastShapes(
    getTensorShape(input1()), getTensorShape(input2()), &params);

  if (need_broadcast)
  {
    tflite::reference_ops::BroadcastAdd4DSlow(
      params, getTensorShape(input1()), getTensorData<T>(input1()), getTensorShape(input2()),
      getTensorData<T>(input2()), getTensorShape(output()), getTensorData<T>(output()));
  }
  else
  {
    tflite::reference_ops::Add(params, getTensorShape(input1()), getTensorData<T>(input1()),
                               getTensorShape(input2()), getTensorData<T>(input2()),
                               getTensorShape(output()), getTensorData<T>(output()));
  }
}

void Add::evalQuantized() const
{
  const auto input1_scale = static_cast<double>(input1()->scale());
  const auto input2_scale = static_cast<double>(input2()->scale());
  const auto output_scale = static_cast<double>(output()->scale());

  const int left_shift = 20;
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
  calculateActivationRangeQuantized(_params.activation, output(), &activation_min, &activation_max);

  tflite::ArithmeticParams params{};
  params.left_shift = left_shift;
  // The kernel expects inputs' zero points to be negated.
  params.input1_offset = -input1()->zero_point(); // Note the '-'.
  params.input1_multiplier = input1_multiplier;
  params.input1_shift = input1_shift;
  params.input2_offset = -input2()->zero_point(); // Note the '-'.
  params.input2_multiplier = input2_multiplier;
  params.input2_shift = input2_shift;
  params.output_offset = output()->zero_point();
  params.output_multiplier = output_multiplier;
  params.output_shift = output_shift;
  params.quantized_activation_min = activation_min;
  params.quantized_activation_max = activation_max;

  const bool need_broadcast = tflite::reference_ops::ProcessBroadcastShapes(
    getTensorShape(input1()), getTensorShape(input2()), &params);

  if (need_broadcast)
  {
    tflite::reference_ops::BroadcastAdd4DSlow(
      params, getTensorShape(input1()), getTensorData<uint8_t>(input1()), getTensorShape(input2()),
      getTensorData<uint8_t>(input2()), getTensorShape(output()), getTensorData<uint8_t>(output()));
  }
  else
  {
    tflite::reference_ops::Add(params, getTensorShape(input1()), getTensorData<uint8_t>(input1()),
                               getTensorShape(input2()), getTensorData<uint8_t>(input2()),
                               getTensorShape(output()), getTensorData<uint8_t>(output()));
  }
}

void Add::evalQuantizedS16() const
{
  const auto input1_scale = static_cast<double>(input1()->scale());
  const auto input2_scale = static_cast<double>(input2()->scale());
  const auto output_scale = static_cast<double>(output()->scale());

  constexpr int left_shift = 12;
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
  calculateActivationRangeQuantized(_params.activation, output(), &activation_min, &activation_max);

  auto fn = [input1_multiplier, input1_shift, //
             input2_multiplier, input2_shift, //
             output_multiplier, output_shift, //
             activation_min, activation_max](int16_t input1_val, int16_t input2_val) {
    const int32_t shifted_input1_val = static_cast<int32_t>(input1_val) << left_shift;
    const int32_t shifted_input2_val = static_cast<int32_t>(input2_val) << left_shift;
    const int32_t scaled_input1_val = tflite::MultiplyByQuantizedMultiplierSmallerThanOneExp(
      shifted_input1_val, input1_multiplier, input1_shift);
    const int32_t scaled_input2_val = tflite::MultiplyByQuantizedMultiplierSmallerThanOneExp(
      shifted_input2_val, input2_multiplier, input2_shift);
    const int32_t raw_sum = scaled_input1_val + scaled_input2_val;
    const int32_t raw_output = tflite::MultiplyByQuantizedMultiplierSmallerThanOneExp(
      raw_sum, output_multiplier, output_shift);
    const int32_t clamped_output = std::min(activation_max, std::max(activation_min, raw_output));
    return static_cast<int16_t>(clamped_output);
  };

  BinaryOpBroadcastSlow(getTensorShape(input1()), getTensorData<int16_t>(input1()),
                        getTensorShape(input2()), getTensorData<int16_t>(input2()),
                        getTensorShape(output()), getTensorData<int16_t>(output()), fn);
}

} // namespace kernels
} // namespace luci_interpreter
