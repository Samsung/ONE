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
#include "cker/operation/optimized/DepthwiseConvFloat.h"
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
  assert(input_shape.DimensionsCount() == 4);
  assert(filter_shape.DimensionsCount() == 4);
  assert(output_shape.DimensionsCount() == 4);

  // Port for thread_count == 1 only
  optimized::DepthwiseConvImpl(params, input_shape, input_data, filter_shape, filter_data,
                               bias_shape, bias_data, output_shape, output_data, 0,
                               output_shape.Dims(1), 1);
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_DEPTHWISE_CONV_H__
