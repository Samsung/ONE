/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/AveragePool2D.h"

#include "kernels/Utils.h"

#include "PALAveragePool2d.h"

#include <stdexcept>

namespace luci_interpreter
{

namespace kernels
{

AveragePool2D::AveragePool2D(const Tensor *input, Tensor *output, Tensor *scratchpad,
                             const Pool2DParams &params)
  : KernelWithParams<Pool2DParams>({input}, {output, scratchpad}, params)
{
}

void AveragePool2D::configure()
{
  if (input()->element_type() != output()->element_type())
  {
    throw std::runtime_error("Input Tensor and Output Tensor Type must be same");
  }
  if (input()->shape().num_dims() != 4)
  {
    throw std::runtime_error("Input Tensor Shape must be 4-D");
  }
  const Shape &input_shape = input()->shape();

  const int32_t batches = input_shape.dim(0);
  const int32_t input_height = input_shape.dim(1);
  const int32_t input_width = input_shape.dim(2);
  const int32_t depth = input_shape.dim(3);

  const int32_t output_height =
    computeOutputSize(_params.padding, input_height, _params.filter_height, _params.stride_height);
  const int32_t output_width =
    computeOutputSize(_params.padding, input_width, _params.filter_width, _params.stride_width);

  _padding_height =
    computePadding(_params.stride_height, 1, input_height, _params.filter_height, output_height);
  _padding_width =
    computePadding(_params.stride_width, 1, input_width, _params.filter_width, output_width);
  if (input()->element_type() == DataType::U8)
  {
    LUCI_INTERPRETER_CHECK(std::abs(output()->scale() - input()->scale()) <= 1.0e-6);
    LUCI_INTERPRETER_CHECK(output()->zero_point() == input()->zero_point());
  }
  else if (input()->element_type() == DataType::S16)
  {
    LUCI_INTERPRETER_CHECK(std::abs(output()->scale() - input()->scale()) <= 1.0e-6);
    LUCI_INTERPRETER_CHECK(input()->zero_point() == 0 && output()->zero_point() == 0);
  }
  else if (input()->element_type() == DataType::S8)
  {
    LUCI_INTERPRETER_CHECK(std::abs(output()->scale() - input()->scale()) <= 1.0e-6);
    LUCI_INTERPRETER_CHECK(output()->zero_point() == input()->zero_point());
  }
  output()->resize({batches, output_height, output_width, depth});

  auto scratchpad = getOutputTensors()[1];
  luci_interpreter_pal::SetupScratchpadTensor(scratchpad, input()->element_type(),
                                              getTensorShape(input()), getTensorShape(output()));
}

void AveragePool2D::execute() const
{
  switch (input()->element_type())
  {
    case DataType::FLOAT32:
      evalFloat();
      break;
    case DataType::U8:
      evalQuantized();
      break;
    case DataType::S16:
      evalSInt16();
      break;
    case DataType::S8:
      evalSInt8();
      break;
    default:
      throw std::runtime_error("luci-intp AveragePool2D Unsupported type.");
  }
}

void AveragePool2D::evalFloat() const
{
  float activation_min{};
  float activation_max{};
  calculateActivationRange(_params.activation, &activation_min, &activation_max);

  tflite::PoolParams params{};
  params.padding_values.height = _padding_height;
  params.padding_values.width = _padding_width;
  params.stride_height = _params.stride_height;
  params.stride_width = _params.stride_width;
  params.filter_height = _params.filter_height;
  params.filter_width = _params.filter_width;
  params.float_activation_min = activation_min;
  params.float_activation_max = activation_max;

  tflite::reference_ops::AveragePool(params, getTensorShape(input()), getTensorData<float>(input()),
                                     getTensorShape(output()), getTensorData<float>(output()));
}

void AveragePool2D::evalQuantized() const
{
  int32_t activation_min{};
  int32_t activation_max{};
  calculateActivationRangeQuantized(_params.activation, output(), &activation_min, &activation_max);

  tflite::PoolParams params{};
  params.padding_values.height = _padding_height;
  params.padding_values.width = _padding_width;
  params.stride_height = _params.stride_height;
  params.stride_width = _params.stride_width;
  params.filter_height = _params.filter_height;
  params.filter_width = _params.filter_width;
  params.quantized_activation_min = activation_min;
  params.quantized_activation_max = activation_max;

  tflite::reference_ops::AveragePool(params, getTensorShape(input()),
                                     getTensorData<uint8_t>(input()), getTensorShape(output()),
                                     getTensorData<uint8_t>(output()));
}

void AveragePool2D::evalSInt8() const
{
  int32_t activation_min{};
  int32_t activation_max{};
  calculateActivationRangeQuantized(_params.activation, output(), &activation_min, &activation_max);
  tflite::PoolParams params{};
  params.padding_values.height = _padding_height;
  params.padding_values.width = _padding_width;
  params.stride_height = _params.stride_height;
  params.stride_width = _params.stride_width;
  params.filter_height = _params.filter_height;
  params.filter_width = _params.filter_width;
  params.quantized_activation_min = activation_min;
  params.quantized_activation_max = activation_max;

  auto scratchpad = getOutputTensors()[1];
  int8_t *scratchpad_data = nullptr;
  if (scratchpad->is_allocatable())
    scratchpad_data = scratchpad->data<int8_t>();

  luci_interpreter_pal::AveragePool<int8_t>(
    params, getTensorShape(input()), getTensorData<int8_t>(input()), getTensorShape(output()),
    getTensorData<int8_t>(output()), getTensorShape(scratchpad), scratchpad_data);
}

void AveragePool2D::evalSInt16() const
{
  int32_t activation_min{};
  int32_t activation_max{};
  calculateActivationRangeQuantized(_params.activation, output(), &activation_min, &activation_max);

  tflite::PoolParams params{};
  params.padding_values.height = _padding_height;
  params.padding_values.width = _padding_width;
  params.stride_height = _params.stride_height;
  params.stride_width = _params.stride_width;
  params.filter_height = _params.filter_height;
  params.filter_width = _params.filter_width;
  params.quantized_activation_min = activation_min;
  params.quantized_activation_max = activation_max;

  tflite::reference_integer_ops::AveragePool(
    params, getTensorShape(input()), getTensorData<int16_t>(input()), //
    getTensorShape(output()), getTensorData<int16_t>(output()));
}

} // namespace kernels
} // namespace luci_interpreter
