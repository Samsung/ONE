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

#include "kernels/DepthwiseConv2D.h"

#include "kernels/Utils.h"

#include <stdexcept>

namespace luci_interpreter
{

namespace kernels
{

DepthwiseConv2D::DepthwiseConv2D(const Tensor *input, const Tensor *filter, const Tensor *bias,
                                 Tensor *output, const DepthwiseConv2DParams &params)
    : _input(input), _filter(filter), _bias(bias), _output(output), _params(params)
{
}

void DepthwiseConv2D::configure()
{
  const Shape &input_shape = _input->shape();
  const Shape &filter_shape = _filter->shape();
  const int32_t batches = input_shape.dim(0);
  const int32_t input_height = input_shape.dim(1);
  const int32_t input_width = input_shape.dim(2);
  const int32_t filter_height = filter_shape.dim(1);
  const int32_t filter_width = filter_shape.dim(2);
  const int32_t channels_out = filter_shape.dim(3);

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

  _output->resize({batches, output_height, output_width, channels_out});
}

void DepthwiseConv2D::execute() const
{
  switch (_input->element_type())
  {
    case DataType::FLOAT32:
      if (_filter->element_type() != DataType::FLOAT32)
      {
        throw std::runtime_error("Hybrid quantization is not supported.");
      }
      evalFloat();
      break;
    case DataType::U8:
      evalQuantized();
      break;
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

// https://github.com/tensorflow/tensorflow/blob/v2.2.0-rc3/tensorflow/lite/kernels/internal/reference/depthwiseconv_float.h
void DepthwiseConv2D::evalFloat() const
{
  const auto *input_data = _input->data<float>();
  const auto *filter_data = _filter->data<float>();
  const auto *bias_data = _bias->data<float>();
  auto *output_data = _output->data<float>();

  const Shape &input_shape = _input->shape();
  const Shape &filter_shape = _filter->shape();
  const Shape &output_shape = _output->shape();

  const int32_t batches = input_shape.dim(0);
  const int32_t input_height = input_shape.dim(1);
  const int32_t input_width = input_shape.dim(2);
  const int32_t input_depth = input_shape.dim(3);
  const int32_t filter_height = filter_shape.dim(1);
  const int32_t filter_width = filter_shape.dim(2);
  const int32_t output_height = output_shape.dim(1);
  const int32_t output_width = output_shape.dim(2);

  float activation_min{};
  float activation_max{};
  calculateActivationRange(_params.activation, &activation_min, &activation_max);

  for (int32_t b = 0; b < batches; ++b)
  {
    for (int32_t out_y = 0; out_y < output_height; ++out_y)
    {
      for (int32_t out_x = 0; out_x < output_width; ++out_x)
      {
        for (int32_t in_c = 0; in_c < input_depth; ++in_c)
        {
          for (int32_t m = 0; m < _params.depth_multiplier; ++m)
          {
            const int32_t out_c = m + in_c * _params.depth_multiplier;
            const int32_t in_y_origin = out_y * _params.stride_height - _padding_height;
            const int32_t in_x_origin = out_x * _params.stride_width - _padding_width;
            float sum = 0.0f;
            for (int32_t filter_y = 0; filter_y < filter_height; ++filter_y)
            {
              for (int32_t filter_x = 0; filter_x < filter_width; ++filter_x)
              {
                const int32_t in_y = in_y_origin + _params.dilation_height_factor * filter_y;
                const int32_t in_x = in_x_origin + _params.dilation_width_factor * filter_x;
                if ((in_y >= 0 && in_y < input_height) && (in_x >= 0 && in_x < input_width))
                {
                  const float input_value = input_data[offset(input_shape, b, in_y, in_x, in_c)];
                  const float filter_value =
                      filter_data[offset(filter_shape, 0, filter_y, filter_x, out_c)];
                  sum += input_value * filter_value;
                }
              }
            }
            sum += bias_data[out_c];
            output_data[offset(output_shape, b, out_y, out_x, out_c)] =
                activationFunctionWithMinMax(sum, activation_min, activation_max);
          }
        }
      }
    }
  }
}

// https://github.com/tensorflow/tensorflow/blob/v2.2.0-rc3/tensorflow/lite/kernels/internal/reference/depthwiseconv_uint8.h
void DepthwiseConv2D::evalQuantized() const
{
  const auto *input_data = _input->data<uint8_t>();
  const auto *filter_data = _filter->data<uint8_t>();
  const auto *bias_data = _bias->data<int32_t>();
  auto *output_data = _output->data<uint8_t>();

  const Shape &input_shape = _input->shape();
  const Shape &filter_shape = _filter->shape();
  const Shape &output_shape = _output->shape();

  const int32_t batches = input_shape.dim(0);
  const int32_t input_height = input_shape.dim(1);
  const int32_t input_width = input_shape.dim(2);
  const int32_t input_depth = input_shape.dim(3);
  const int32_t filter_height = filter_shape.dim(1);
  const int32_t filter_width = filter_shape.dim(2);
  const int32_t output_height = output_shape.dim(1);
  const int32_t output_width = output_shape.dim(2);

  const int32_t input_offset = _input->zero_point();
  const int32_t filter_offset = _filter->zero_point();
  const int32_t output_offset = _output->zero_point();

  const double input_scale = _input->scale();
  const double filter_scale = _filter->scale();
  const double output_scale = _output->scale();

  const double real_multiplier = input_scale * filter_scale / output_scale;
  int32_t output_multiplier{};
  int output_shift{};
  quantizeMultiplier(real_multiplier, &output_multiplier, &output_shift);

  int32_t activation_min{};
  int32_t activation_max{};
  calculateActivationRangeQuantized(_params.activation, _output, &activation_min, &activation_max);

  for (int32_t b = 0; b < batches; ++b)
  {
    for (int32_t out_y = 0; out_y < output_height; ++out_y)
    {
      for (int32_t out_x = 0; out_x < output_width; ++out_x)
      {
        for (int32_t in_c = 0; in_c < input_depth; ++in_c)
        {
          for (int32_t m = 0; m < _params.depth_multiplier; ++m)
          {
            const int32_t out_c = m + in_c * _params.depth_multiplier;
            const int32_t in_y_origin = out_y * _params.stride_height - _padding_height;
            const int32_t in_x_origin = out_x * _params.stride_width - _padding_width;
            int32_t sum = 0;
            for (int32_t filter_y = 0; filter_y < filter_height; ++filter_y)
            {
              for (int32_t filter_x = 0; filter_x < filter_width; ++filter_x)
              {
                const int32_t in_y = in_y_origin + _params.dilation_height_factor * filter_y;
                const int32_t in_x = in_x_origin + _params.dilation_width_factor * filter_x;
                if ((in_y >= 0 && in_y < input_height) && (in_x >= 0 && in_x < input_width))
                {
                  const float input_value = input_data[offset(input_shape, b, in_y, in_x, in_c)];
                  const float filter_value =
                      filter_data[offset(filter_shape, 0, filter_y, filter_x, out_c)];
                  sum += (input_value - input_offset) * (filter_value - filter_offset);
                }
              }
            }
            sum += bias_data[out_c];
            sum = multiplyByQuantizedMultiplier(sum, output_multiplier, output_shift);
            sum += output_offset;
            sum = activationFunctionWithMinMax(sum, activation_min, activation_max);
            output_data[offset(output_shape, b, out_y, out_x, out_c)] = static_cast<uint8_t>(sum);
          }
        }
      }
    }
  }
}

} // namespace kernels
} // namespace luci_interpreter
