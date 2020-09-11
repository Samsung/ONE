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

#include "kernels/DepthwiseConv2D.h"

#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/reference/depthwiseconv_float.h>
#include <tensorflow/lite/kernels/internal/reference/depthwiseconv_uint8.h>

#include <stdexcept>

namespace luci_interpreter
{
namespace kernels
{

DepthwiseConv2D::DepthwiseConv2D(const Tensor *input, const Tensor *filter, const Tensor *bias,
                                 Tensor *output, const DepthwiseConv2DParams &params)
    : KernelWithParams<DepthwiseConv2DParams>({input, filter, bias}, {output}, params)
{
}

void DepthwiseConv2D::configure()
{
  // TensorFlow Lite (as of v2.2.0) supports the following combinations of types:
  //     | input filter bias  output |
  // ----+---------------------------+
  // (1) | float float  float float  |
  // (2) | float int8   float float  | hybrid
  // (3) | uint8 uint8  int32 uint8  | quantized
  // (4) | int8  int8   int32 int8   | quantized per channel
  // (5) | int16 int8   int64 int16  | quantized per channel 16x8
  //
  // We only support (1) and (3) for now.
  if (input()->element_type() == DataType::FLOAT32 && filter()->element_type() == DataType::FLOAT32)
  {
    LUCI_INTERPRETER_CHECK(bias() == nullptr || bias()->element_type() == DataType::FLOAT32);
  }
  else if (input()->element_type() == DataType::U8 && filter()->element_type() == DataType::U8)
  {
    LUCI_INTERPRETER_CHECK(bias() == nullptr || bias()->element_type() == DataType::S32);
  }
  else
  {
    throw std::runtime_error("Unsupported type.");
  }
  LUCI_INTERPRETER_CHECK(output()->element_type() == input()->element_type());

  const Shape &input_shape = input()->shape();
  const Shape &filter_shape = filter()->shape();
  LUCI_INTERPRETER_CHECK(input_shape.num_dims() == 4 && filter_shape.num_dims() == 4);

  const int32_t batches = input_shape.dim(0);
  const int32_t input_height = input_shape.dim(1);
  const int32_t input_width = input_shape.dim(2);
  // Filter format: [1, H, W, O].
  LUCI_INTERPRETER_CHECK(filter_shape.dim(0) == 1);
  const int32_t filter_height = filter_shape.dim(1);
  const int32_t filter_width = filter_shape.dim(2);
  const int32_t channels_out = filter_shape.dim(3);

  LUCI_INTERPRETER_CHECK(bias() == nullptr || (bias()->shape().num_dims() == 1 &&
                                               bias()->shape().dim(0) == channels_out));

  const int32_t output_height =
      computeOutputSize(_params.padding, input_height, filter_height, _params.stride_height,
                        _params.dilation_height_factor);
  const int32_t output_width =
      computeOutputSize(_params.padding, input_width, filter_width, _params.stride_width,
                        _params.dilation_width_factor);

  _padding_height = computePadding(_params.stride_height, _params.dilation_height_factor,
                                   input_height, filter_height, output_height);
  _padding_width = computePadding(_params.stride_width, _params.dilation_width_factor, input_width,
                                  filter_width, output_width);

  output()->resize({batches, output_height, output_width, channels_out});
}

void DepthwiseConv2D::execute() const
{
  switch (input()->element_type())
  {
    case DataType::FLOAT32:
      if (filter()->element_type() == DataType::FLOAT32)
      {
        evalFloat();
        break;
      }
      throw std::runtime_error("Unsupported type.");
    case DataType::U8:
      evalQuantized();
      break;
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

void DepthwiseConv2D::evalFloat() const
{
  float activation_min{};
  float activation_max{};
  calculateActivationRange(_params.activation, &activation_min, &activation_max);

  tflite::DepthwiseParams params{};
  params.padding_values.height = _padding_height;
  params.padding_values.width = _padding_width;
  params.stride_height = _params.stride_height;
  params.stride_width = _params.stride_width;
  params.dilation_height_factor = _params.dilation_height_factor;
  params.dilation_width_factor = _params.dilation_width_factor;
  params.depth_multiplier = _params.depth_multiplier;
  params.float_activation_min = activation_min;
  params.float_activation_max = activation_max;

  tflite::reference_ops::DepthwiseConv(
      params, getTensorShape(input()), getTensorData<float>(input()), getTensorShape(filter()),
      getTensorData<float>(filter()), getTensorShape(bias()), getTensorData<float>(bias()),
      getTensorShape(output()), getTensorData<float>(output()));
}

void DepthwiseConv2D::evalQuantized() const
{
  const auto input_scale = static_cast<double>(input()->scale());
  const auto filter_scale = static_cast<double>(filter()->scale());
  const auto output_scale = static_cast<double>(output()->scale());

  const double real_multiplier = input_scale * filter_scale / output_scale;
  int32_t output_multiplier{};
  int output_shift{};
  quantizeMultiplier(real_multiplier, &output_multiplier, &output_shift);

  int32_t activation_min{};
  int32_t activation_max{};
  calculateActivationRangeQuantized(_params.activation, output(), &activation_min, &activation_max);

  tflite::DepthwiseParams params{};
  params.padding_values.height = _padding_height;
  params.padding_values.width = _padding_width;
  params.stride_height = _params.stride_height;
  params.stride_width = _params.stride_width;
  params.dilation_height_factor = _params.dilation_height_factor;
  params.dilation_width_factor = _params.dilation_width_factor;
  params.depth_multiplier = _params.depth_multiplier;
  // The kernel expects input and filter zero points to be negated.
  params.input_offset = -input()->zero_point();    // Note the '-'.
  params.weights_offset = -filter()->zero_point(); // Note the '-'.
  params.output_offset = output()->zero_point();
  params.output_multiplier = output_multiplier;
  params.output_shift = output_shift;
  params.quantized_activation_min = activation_min;
  params.quantized_activation_max = activation_max;

  tflite::reference_ops::DepthwiseConv(
      params, getTensorShape(input()), getTensorData<uint8_t>(input()), getTensorShape(filter()),
      getTensorData<uint8_t>(filter()), getTensorShape(bias()), getTensorData<int32_t>(bias()),
      getTensorShape(output()), getTensorData<uint8_t>(output()));
}

} // namespace kernels
} // namespace luci_interpreter
