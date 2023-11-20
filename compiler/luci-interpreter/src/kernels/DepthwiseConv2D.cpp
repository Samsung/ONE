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

#include "PALDepthwiseConv2d.h"

#include <stdexcept>

namespace luci_interpreter
{
namespace kernels
{

DepthwiseConv2D::DepthwiseConv2D(const Tensor *input, const Tensor *filter, const Tensor *bias,
                                 Tensor *output, Tensor *scratchpad,
                                 const DepthwiseConv2DParams &params)
  : KernelWithParams<DepthwiseConv2DParams>({input, filter, bias}, {output, scratchpad}, params)
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
  // We only support (1), (3) and (4) for now, and additionally the following:
  //     | input filter bias  output |
  // ----+---------------------------+
  // (5) | int16 int16  int64 int16  |
  //
  if (input()->element_type() == DataType::FLOAT32 && filter()->element_type() == DataType::FLOAT32)
  {
    LUCI_INTERPRETER_CHECK(bias() == nullptr || bias()->element_type() == DataType::FLOAT32);
  }
  else if (input()->element_type() == DataType::U8 && filter()->element_type() == DataType::U8)
  {
    LUCI_INTERPRETER_CHECK(bias() == nullptr || bias()->element_type() == DataType::S32);
  }
  else if (input()->element_type() == DataType::S8 && filter()->element_type() == DataType::S8)
  {
    LUCI_INTERPRETER_CHECK(filter()->shape().num_dims() == 4);
    LUCI_INTERPRETER_CHECK(static_cast<uint32_t>(filter()->shape().dim(3)) ==
                           filter()->scales().size());
    for (auto zerop : filter()->zero_points())
    {
      LUCI_INTERPRETER_CHECK(zerop == 0);
    }
    LUCI_INTERPRETER_CHECK(bias() == nullptr || bias()->element_type() == DataType::S32);
  }
  else if (input()->element_type() == DataType::S16 && filter()->element_type() == DataType::S16)
  {
    LUCI_INTERPRETER_CHECK(bias() == nullptr || bias()->element_type() == DataType::S64);
  }
  else
  {
    throw std::runtime_error("luci-intp DepthwiseConv2D(1) Unsupported type.");
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

  tflite::DepthwiseParams params{};

  params.dilation_height_factor = _params.dilation_height_factor;
  params.dilation_width_factor = _params.dilation_width_factor;

  auto scratchpad = getOutputTensors()[1];
  luci_interpreter_pal::SetupScratchpadTensor(scratchpad, params, input()->element_type(),
                                              getTensorShape(input()), getTensorShape(filter()),
                                              getTensorShape(output()));
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
      throw std::runtime_error("luci-intp DepthwiseConv2D(2) Unsupported type.");
    case DataType::U8:
      if (filter()->scales().size() == 1)
      {
        evalQuantized();
      }
      else if (filter()->scales().size() > 1)
      {
        LUCI_INTERPRETER_CHECK(filter()->shape().num_dims() == 4);
        LUCI_INTERPRETER_CHECK(filter()->scales().size() ==
                               static_cast<size_t>(filter()->shape().dim(3)));
        evalQuantizedPerChannel();
      }
      break;
    case DataType::S8:
      evalQuantizedS8PerChannel();
      break;
    case DataType::S16:
      evalQuantizedS16();
      break;
    default:
      throw std::runtime_error("luci-intp DepthwiseConv2D(3) Unsupported type.");
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

void DepthwiseConv2D::evalQuantizedPerChannel() const
{
  const auto *input_data = getTensorData<uint8_t>(input());
  const auto *filter_data = getTensorData<uint8_t>(filter());
  const auto *bias_data = getTensorData<int32_t>(bias());
  auto *output_data = getTensorData<uint8_t>(output());

  const Shape &input_shape = input()->shape();
  const Shape &filter_shape = filter()->shape();
  const Shape &output_shape = output()->shape();

  const int32_t batches = input_shape.dim(0);
  const int32_t input_height = input_shape.dim(1);
  const int32_t input_width = input_shape.dim(2);
  const int32_t input_depth = input_shape.dim(3);
  const int32_t filter_height = filter_shape.dim(1);
  const int32_t filter_width = filter_shape.dim(2);
  const int32_t output_height = output_shape.dim(1);
  const int32_t output_width = output_shape.dim(2);

  const int32_t stride_height = _params.stride_height;
  const int32_t stride_width = _params.stride_width;
  const int32_t dilation_height_factor = _params.dilation_height_factor;
  const int32_t dilation_width_factor = _params.dilation_width_factor;
  const int32_t depth_multiplier = _params.depth_multiplier;

  int32_t activation_min{};
  int32_t activation_max{};
  calculateActivationRangeQuantized(_params.activation, output(), &activation_min, &activation_max);

  const std::vector<double> effective_output_scales =
    getQuantizedConvolutionMultiplers(input()->scale(), filter()->scales(), output()->scale());

  std::vector<ChannelQuantMultipliers> quant_multipliers_raw =
    quantizeMultipliers(effective_output_scales);
  BroadcastableWrapper<ChannelQuantMultipliers> quant_multipliers(quant_multipliers_raw);

  for (int batch = 0; batch < batches; ++batch)
  {
    for (int out_y = 0; out_y < output_height; ++out_y)
    {
      for (int out_x = 0; out_x < output_width; ++out_x)
      {
        for (int in_channel = 0; in_channel < input_depth; ++in_channel)
        {
          for (int m = 0; m < depth_multiplier; ++m)
          {
            const int output_channel = m + in_channel * depth_multiplier;
            const int in_x_origin = (out_x * stride_width) - _padding_width;
            const int in_y_origin = (out_y * stride_height) - _padding_height;
            int32 acc = 0;
            for (int filter_y = 0; filter_y < filter_height; ++filter_y)
            {
              for (int filter_x = 0; filter_x < filter_width; ++filter_x)
              {
                const int in_x = in_x_origin + dilation_width_factor * filter_x;
                const int in_y = in_y_origin + dilation_height_factor * filter_y;
                // Zero padding by omitting the areas outside the image.
                const bool is_point_inside_image =
                  (in_x >= 0) && (in_x < input_width) && (in_y >= 0) && (in_y < input_height);
                if (is_point_inside_image)
                {
                  int32 input_val =
                    input_data[calcOffset(input_shape, batch, in_y, in_x, in_channel)];
                  int32 filter_val =
                    filter_data[calcOffset(filter_shape, 0, filter_y, filter_x, output_channel)];
                  acc += (filter_val - filter()->zero_points()[output_channel]) *
                         (input_val - input()->zero_point());
                }
              }
            }
            if (bias_data)
            {
              acc += bias_data[output_channel];
            }
            int32_t output_multiplier = quant_multipliers[output_channel].multiplier;
            int output_shift = quant_multipliers[output_channel].shift;
            int32_t scaled_acc =
              tflite::MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
            scaled_acc += output()->zero_point();
            scaled_acc = std::max(scaled_acc, activation_min);
            scaled_acc = std::min(scaled_acc, activation_max);
            output_data[calcOffset(output_shape, batch, out_y, out_x, output_channel)] =
              static_cast<uint8_t>(scaled_acc);
          }
        }
      }
    }
  }
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

void DepthwiseConv2D::evalQuantizedS8PerChannel() const
{
  int32_t activation_min{};
  int32_t activation_max{};
  calculateActivationRangeQuantized(_params.activation, output(), &activation_min, &activation_max);

  tflite::DepthwiseParams params{};

  params.padding_type = tflite::PaddingType::kSame;
  params.padding_values.height = _padding_height;
  params.padding_values.width = _padding_width;
  params.stride_height = _params.stride_height;
  params.stride_width = _params.stride_width;
  params.dilation_height_factor = _params.dilation_height_factor;
  params.dilation_width_factor = _params.dilation_width_factor;
  params.depth_multiplier = _params.depth_multiplier;
  // The kernel expects input and filter zero points to be negated.
  params.input_offset = -input()->zero_point(); // Note the '-'.
  params.weights_offset = 0;
  params.output_offset = output()->zero_point();
  params.output_multiplier = 1; // unused in tflite code
  params.output_shift = 0;      // unused in tflite code
  params.quantized_activation_min = activation_min;
  params.quantized_activation_max = activation_max;

  const std::vector<double> effective_output_scales =
    getQuantizedConvolutionMultiplers(input()->scale(), filter()->scales(), output()->scale());

  std::vector<ChannelQuantMultipliers> quant_multipliers =
    quantizeMultipliers(effective_output_scales);

  std::vector<int32_t> shifts;
  std::transform(quant_multipliers.begin(), quant_multipliers.end(), std::back_inserter(shifts),
                 [](ChannelQuantMultipliers cm) { return cm.shift; });
  std::vector<int32_t> multipliers;
  std::transform(quant_multipliers.begin(), quant_multipliers.end(),
                 std::back_inserter(multipliers),
                 [](ChannelQuantMultipliers cm) { return cm.multiplier; });

  auto scratchpad = getOutputTensors()[1];
  int8_t *scratchpad_data = nullptr;
  if (scratchpad->is_allocatable())
    scratchpad_data = scratchpad->data<int8_t>();

  luci_interpreter_pal::DepthwiseConvPerChannel<int8_t>(
    params, multipliers.data(), shifts.data(), getTensorShape(input()),
    getTensorData<int8_t>(input()), getTensorShape(filter()), getTensorData<int8_t>(filter()),
    getTensorShape(bias()), getTensorData<int32_t>(bias()), getTensorShape(output()),
    getTensorData<int8_t>(output()), getTensorShape(scratchpad), scratchpad_data);
}

void DepthwiseConv2D::evalQuantizedS16() const
{
  const auto *input_data = getTensorData<int16_t>(input());
  const auto *filter_data = getTensorData<int16_t>(filter());
  const auto *bias_data = getTensorData<int64_t>(bias());
  auto *output_data = getTensorData<int16_t>(output());

  const Shape &input_shape = input()->shape();
  const Shape &filter_shape = filter()->shape();
  const Shape &output_shape = output()->shape();

  const int32_t batches = input_shape.dim(0);
  const int32_t input_height = input_shape.dim(1);
  const int32_t input_width = input_shape.dim(2);
  const int32_t input_depth = input_shape.dim(3);
  const int32_t filter_height = filter_shape.dim(1);
  const int32_t filter_width = filter_shape.dim(2);
  const int32_t output_height = output_shape.dim(1);
  const int32_t output_width = output_shape.dim(2);

  const int32_t stride_height = _params.stride_height;
  const int32_t stride_width = _params.stride_width;
  const int32_t dilation_height_factor = _params.dilation_height_factor;
  const int32_t dilation_width_factor = _params.dilation_width_factor;
  const int32_t depth_multiplier = _params.depth_multiplier;

  const std::vector<double> effective_output_scales =
    getQuantizedConvolutionMultiplers(input()->scale(), filter()->scales(), output()->scale());

  std::vector<ChannelQuantMultipliers> quant_multipliers_raw =
    quantizeMultipliers(effective_output_scales);

  BroadcastableWrapper<ChannelQuantMultipliers> quant_multipliers(quant_multipliers_raw);

  int32_t activation_min{};
  int32_t activation_max{};
  calculateActivationRangeQuantized(_params.activation, output(), &activation_min, &activation_max);

  for (int32_t batch = 0; batch < batches; ++batch)
  {
    for (int32_t out_y = 0; out_y < output_height; ++out_y)
    {
      for (int32_t out_x = 0; out_x < output_width; ++out_x)
      {
        for (int32_t in_c = 0; in_c < input_depth; ++in_c)
        {
          for (int32_t m = 0; m < depth_multiplier; ++m)
          {
            const int32_t out_c = m + in_c * depth_multiplier;
            const int32_t in_y_origin = out_y * stride_height - _padding_height;
            const int32_t in_x_origin = out_x * stride_width - _padding_width;
            int64_t acc = 0;
            for (int32_t filter_y = 0; filter_y < filter_height; ++filter_y)
            {
              for (int32_t filter_x = 0; filter_x < filter_width; ++filter_x)
              {
                const int32_t in_y = in_y_origin + dilation_height_factor * filter_y;
                const int32_t in_x = in_x_origin + dilation_width_factor * filter_x;
                if ((in_y >= 0 && in_y < input_height) && (in_x >= 0 && in_x < input_width))
                {
                  const int16_t input_val =
                    input_data[calcOffset(input_shape, batch, in_y, in_x, in_c)];
                  const int16_t filter_val =
                    filter_data[calcOffset(filter_shape, 0, filter_y, filter_x, out_c)];
                  acc += static_cast<int64_t>(input_val) * static_cast<int64_t>(filter_val);
                }
              }
            }
            if (bias_data != nullptr)
            {
              acc += bias_data[out_c];
            }

            int32_t output_multiplier = quant_multipliers[out_c].multiplier;
            int output_shift = quant_multipliers[out_c].shift;
            int32_t scaled_acc =
              tflite::MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);

            scaled_acc = std::max(scaled_acc, activation_min);
            scaled_acc = std::min(scaled_acc, activation_max);

            output_data[calcOffset(output_shape, batch, out_y, out_x, out_c)] = scaled_acc;
          }
        }
      }
    }
  }
}

} // namespace kernels
} // namespace luci_interpreter
