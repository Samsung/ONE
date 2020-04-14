/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __NNFW_CKER_DEPTHWISE_CONV_H__
#define __NNFW_CKER_DEPTHWISE_CONV_H__

#include "cker/Shape.h"
#include "cker/Types.h"
#include "cker/Utils.h"
#include "cker/neon/neon_check.h"
#include "cker/operation/optimized/DepthwiseConvUint8.h"

namespace nnfw
{
namespace cker
{

inline void DepthwiseConv(const DepthwiseConvParams &params, const Shape &input_shape,
                          const uint8_t *input_data, const Shape &filter_shape,
                          const uint8_t *filter_data, const Shape &bias_shape,
                          const int32_t *bias_data, const Shape &output_shape, uint8_t *output_data)
{
  const int depth_multiplier = params.depth_multiplier;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  assert(dilation_width_factor >= 1);
  assert(dilation_height_factor >= 1);
  UNUSED_RELEASE(dilation_width_factor);
  UNUSED_RELEASE(dilation_height_factor);
  assert(input_shape.DimensionsCount() == 4);
  assert(filter_shape.DimensionsCount() == 4);
  assert(output_shape.DimensionsCount() == 4);
  assert(output_activation_min <= output_activation_max);
  UNUSED_RELEASE(output_activation_min);
  UNUSED_RELEASE(output_activation_max);
  const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);
  const int input_depth = input_shape.Dims(3);
  assert(output_depth == input_depth * depth_multiplier);
  assert(bias_shape.FlatSize() == output_depth);
  UNUSED_RELEASE(input_depth);
  UNUSED_RELEASE(output_depth);
  UNUSED_RELEASE(depth_multiplier);

// Enable for arm64 except for the Nvidia Linux 4 Tegra (L4T) running on
// Jetson TX-2. This compiler does not support the offsetof() macro.
#if defined(__aarch64__)
//  TODO Use below codes

//  const int stride_width = params.stride_width;
//  const int stride_height = params.stride_height;
//  const int pad_width = params.padding_values.width;
//  const int pad_height = params.padding_values.height;
//  const int output_shift = params.output_shift;
//
//  // Call kernel optimized for depthwise convolutions using 3x3 filters if
//  // parameters are supported.
//  if (Fast3x3FilterKernelSupported(
//          input_shape, filter_shape, stride_width, stride_height,
//          dilation_width_factor, dilation_height_factor, pad_width, pad_height,
//          depth_multiplier, output_shape, output_shift)) {
//    DepthwiseConv3x3Filter(params, input_shape, input_data, filter_shape,
//                           filter_data, bias_shape, bias_data, output_shape,
//                           output_data);
//    return;
//  }
#endif

  optimized::DepthwiseConvGeneral(params, input_shape, input_data, filter_shape, filter_data,
                                  bias_shape, bias_data, output_shape, output_data);
}

inline void DepthwiseConv(const DepthwiseConvParams &params, const Shape &input_shape,
                          const float *input_data, const Shape &filter_shape,
                          const float *filter_data, const Shape &bias_shape, const float *bias_data,
                          const Shape &output_shape, float *output_data)
{
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int depth_multiplier = params.depth_multiplier;
  const float output_activation_min = params.float_activation_min;
  const float output_activation_max = params.float_activation_max;
  assert(input_shape.DimensionsCount() == 4);
  assert(filter_shape.DimensionsCount() == 4);
  assert(output_shape.DimensionsCount() == 4);

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth = input_shape.Dims(3);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  assert(output_depth == input_depth * depth_multiplier);
  assert(bias_shape.FlatSize() == output_depth);
  UNUSED_RELEASE(output_depth);
  UNUSED_RELEASE(bias_shape);

  for (int b = 0; b < batches; ++b)
  {
    for (int out_y = 0; out_y < output_height; ++out_y)
    {
      for (int out_x = 0; out_x < output_width; ++out_x)
      {
        for (int ic = 0; ic < input_depth; ++ic)
        {
          for (int m = 0; m < depth_multiplier; m++)
          {
            const int oc = m + ic * depth_multiplier;
            const int in_x_origin = (out_x * stride_width) - pad_width;
            const int in_y_origin = (out_y * stride_height) - pad_height;
            float total = 0.f;
            for (int filter_y = 0; filter_y < filter_height; ++filter_y)
            {
              for (int filter_x = 0; filter_x < filter_width; ++filter_x)
              {
                const int in_x = in_x_origin + dilation_width_factor * filter_x;
                const int in_y = in_y_origin + dilation_height_factor * filter_y;
                // If the location is outside the bounds of the input image,
                // use zero as a default value.
                if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) && (in_y < input_height))
                {
                  float input_value = input_data[Offset(input_shape, b, in_y, in_x, ic)];
                  float filter_value = filter_data[Offset(filter_shape, 0, filter_y, filter_x, oc)];
                  total += (input_value * filter_value);
                }
              }
            }
            float bias_value = 0.0f;
            if (bias_data)
            {
              bias_value = bias_data[oc];
            }
            output_data[Offset(output_shape, b, out_y, out_x, oc)] = ActivationFunctionWithMinMax(
                total + bias_value, output_activation_min, output_activation_max);
          }
        }
      }
    }
  }
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_DEPTHWISE_CONV_H__
