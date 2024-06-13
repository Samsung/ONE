/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "execute/kernels/ConvolutionCommon.h"
#include "execute/OMUtils.h"

using namespace onert_micro;
using namespace onert_micro::core;

OMStatus onert_micro::execute::createConvParams(core::ConvQuant &params,
                                                const circle::Tensor *input,
                                                const circle::Tensor *filter,
                                                const circle::Tensor *output,
                                                const circle::Conv2DOptions *options)
{
  const auto padding = options->padding();
  const auto stride_height = options->stride_h();
  const auto stride_width = options->stride_w();
  const auto dilation_height_factor = options->dilation_h_factor();
  const auto dilation_width_factor = options->dilation_h_factor();

  params.stride_height = stride_height;
  params.stride_width = stride_width;
  params.dilation_height_factor = dilation_height_factor;
  params.dilation_width_factor = dilation_width_factor;

  assert(input->quantization() != nullptr);  // Fix caller
  assert(filter->quantization() != nullptr); // Fix caller
  assert(output->quantization() != nullptr); // Fix caller

  const auto *input_scales = input->quantization()->scale();
  const auto *filter_scales = filter->quantization()->scale();
  const auto *output_scales = output->quantization()->scale();

  assert(input_scales != nullptr);  // Fix caller
  assert(filter_scales != nullptr); // Fix caller
  assert(output_scales != nullptr); // Fix caller

  assert(input_scales->size() != 0);  // Fix caller
  assert(filter_scales->size() != 0); // Fix caller
  assert(output_scales->size() != 0); // Fix caller

  const auto input_zero_points = input->quantization()->zero_point();
  const auto filter_zero_points = filter->quantization()->zero_point();
  const auto output_zero_points = output->quantization()->zero_point();

  assert(input_zero_points != nullptr);  // Fix caller
  assert(filter_zero_points != nullptr); // Fix caller
  assert(output_zero_points != nullptr); // Fix caller

  assert(input_zero_points->size() != 0);  // Fix caller
  assert(filter_zero_points->size() != 0); // Fix caller
  assert(output_zero_points->size() != 0); // Fix caller

  const auto input_zp = input_zero_points->operator[](0);
  const auto filter_zp = filter_zero_points->operator[](0);
  const auto output_zp = output_zero_points->operator[](0);

  const auto output_scale = output_scales->operator[](0);

  int32_t activation_min{};
  int32_t activation_max{};
  OMStatus status = execute::calculateActivationRangeQuantized(
    options->fused_activation_function(), static_cast<int32_t>(output_zp), output_scale,
    output->type(), &activation_min, &activation_max);
  assert(status == Ok);
  if (status != Ok)
    return status;

  // The kernel expects input and filter zero points to be negated.
  params.input_offset = -static_cast<int32_t>(input_zp);    // Note the '-'.
  params.weights_offset = -static_cast<int32_t>(filter_zp); // Note the '-'.
  params.output_offset = static_cast<int32_t>(output_zp);
  params.quantized_activation_min = activation_min;
  params.quantized_activation_max = activation_max;

  assert(filter_scales->size() > 1); // Support only channel-wise quantization
  // Channel-wise quantization
  const auto input_scale = input_scales->operator[](0);
  const std::vector<double> effective_output_scale =
    execute::getQuantizedConvolutionMultiplers(input_scale, filter_scales, output_scale);

  size_t n = effective_output_scale.size();
  params.per_channel_output_shift.resize(n);
  params.per_channel_output_multiplier.resize(n);
  for (size_t i = 0; i < n; ++i)
  {
    execute::quantizeMultiplier(effective_output_scale[i], &params.per_channel_output_multiplier[i],
                                &params.per_channel_output_shift[i]);
  }

  return Ok;
}
