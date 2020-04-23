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

#include "kernels/Conv2D.h"

#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/reference/conv.h>

#include <stdexcept>

namespace luci_interpreter
{

namespace kernels
{

Conv2D::Conv2D(const Tensor *input, const Tensor *filter, const Tensor *bias, Tensor *output,
               const Conv2DParams &params)
    : _input(input), _filter(filter), _bias(bias), _output(output), _params(params)
{
}

void Conv2D::configure()
{
  const Shape &input_shape = _input->shape();
  const Shape &filter_shape = _filter->shape();

  const int32_t batches = input_shape.dim(0);
  const int32_t input_height = input_shape.dim(1);
  const int32_t input_width = input_shape.dim(2);
  const int32_t output_depth = filter_shape.dim(0);
  const int32_t filter_height = filter_shape.dim(1);
  const int32_t filter_width = filter_shape.dim(2);

  const int32_t dilation_height_factor = 1;
  const int32_t dilation_width_factor = 1;

  const int32_t output_height = computeOutputSize(_params.padding, input_height, filter_height,
                                                  _params.stride_height, dilation_height_factor);
  const int32_t output_width = computeOutputSize(_params.padding, input_width, filter_width,
                                                 _params.stride_width, dilation_width_factor);

  _padding_height = computePadding(_params.stride_height, dilation_height_factor, input_height,
                                   filter_height, output_height);
  _padding_width = computePadding(_params.stride_width, dilation_width_factor, input_width,
                                  filter_width, output_width);

  _output->resize({batches, output_height, output_width, output_depth});
}

void Conv2D::execute() const
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

void Conv2D::evalFloat() const
{
  float activation_min{};
  float activation_max{};
  calculateActivationRange(_params.activation, &activation_min, &activation_max);

  tflite::ConvParams params{};
  params.padding_values.height = _padding_height;
  params.padding_values.width = _padding_width;
  params.stride_height = _params.stride_height;
  params.stride_width = _params.stride_width;
  params.dilation_height_factor = _params.dilation_height_factor;
  params.dilation_width_factor = _params.dilation_width_factor;
  params.float_activation_min = activation_min;
  params.float_activation_max = activation_max;

  tflite::reference_ops::Conv(
      params, convertShape(_input->shape()), _input->data<float>(), convertShape(_filter->shape()),
      _filter->data<float>(), convertShape(_bias->shape()), _bias->data<float>(),
      convertShape(_output->shape()), _output->data<float>(), tflite::RuntimeShape(), nullptr);
}

void Conv2D::evalQuantized() const
{
  const auto input_scale = static_cast<double>(_input->scale());
  const auto filter_scale = static_cast<double>(_filter->scale());
  const auto output_scale = static_cast<double>(_output->scale());

  const double real_multiplier = input_scale * filter_scale / output_scale;
  int32_t output_multiplier{};
  int output_shift{};
  quantizeMultiplier(real_multiplier, &output_multiplier, &output_shift);

  int32_t activation_min{};
  int32_t activation_max{};
  calculateActivationRangeQuantized(_params.activation, _output, &activation_min, &activation_max);

  tflite::ConvParams params{};
  params.padding_values.height = _padding_height;
  params.padding_values.width = _padding_width;
  params.stride_height = _params.stride_height;
  params.stride_width = _params.stride_width;
  params.dilation_height_factor = _params.dilation_height_factor;
  params.dilation_width_factor = _params.dilation_width_factor;
  params.input_offset = -_input->zero_point();    // Note the '-'.
  params.weights_offset = -_filter->zero_point(); // Note the '-'.
  params.output_offset = _output->zero_point();
  params.output_multiplier = output_multiplier;
  params.output_shift = output_shift;
  params.quantized_activation_min = activation_min;
  params.quantized_activation_max = activation_max;

  tflite::reference_ops::Conv(params, convertShape(_input->shape()), _input->data<uint8_t>(),
                              convertShape(_filter->shape()), _filter->data<uint8_t>(),
                              convertShape(_bias->shape()), _bias->data<int32_t>(),
                              convertShape(_output->shape()), _output->data<uint8_t>(),
                              tflite::RuntimeShape(), nullptr, nullptr);
}

} // namespace kernels
} // namespace luci_interpreter
