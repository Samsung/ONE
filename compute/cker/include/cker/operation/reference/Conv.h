/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __NNFW_CKER_REFERENCE_CONV_H__
#define __NNFW_CKER_REFERENCE_CONV_H__

#include "cker/Shape.h"
#include "cker/Types.h"

#include <cmath>

namespace nnfw
{
namespace cker
{
namespace reference
{

inline void Conv(const ConvParams &params, const Shape &input_shape, const float *input_data,
                 const Shape &filter_shape, const float *filter_data, const Shape &bias_shape,
                 const float *bias_data, const Shape &output_shape, float *output_data)
{
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const float output_activation_min = params.float_activation_min;
  const float output_activation_max = params.float_activation_max;
  assert(input_shape.DimensionsCount() == 4);
  assert(filter_shape.DimensionsCount() == 4);
  assert(output_shape.DimensionsCount() == 4);
  UNUSED_RELEASE(bias_shape);

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data)
  {
    assert(bias_shape.FlatSize() == output_depth);
  }
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  for (int batch = 0; batch < batches; ++batch)
  {
    for (int out_y = 0; out_y < output_height; ++out_y)
    {
      for (int out_x = 0; out_x < output_width; ++out_x)
      {
        for (int out_channel = 0; out_channel < output_depth; ++out_channel)
        {
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
                const int in_offset = Offset(input_shape, batch, in_y, in_x, 0);
                const int filter_offset = Offset(filter_shape, out_channel, filter_y, filter_x, 0);
                for (int in_channel = 0; in_channel < input_depth; ++in_channel)
                {
                  float input_value = input_data[in_offset + in_channel];
                  float filter_value = filter_data[filter_offset + in_channel];
                  total += (input_value * filter_value);
                }
              }
            }
          }
          float bias_value = 0.0f;
          if (bias_data)
          {
            bias_value = bias_data[out_channel];
          }
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
            ActivationFunctionWithMinMax(total + bias_value, output_activation_min,
                                         output_activation_max);
        }
      }
    }
  }
}

inline void Conv(const ConvParams &params, const Shape &input_shape, const uint8_t *input_data,
                 const Shape &filter_shape, const uint8_t *filter_data, const Shape &bias_shape,
                 const int32_t *bias_data, const Shape &output_shape, uint8_t *output_data)
{
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t input_offset = params.input_offset;
  const int32_t filter_offset = params.weights_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  assert(output_activation_min <= output_activation_max);

  assert(input_shape.DimensionsCount() == 4);
  assert(filter_shape.DimensionsCount() == 4);
  assert(output_shape.DimensionsCount() == 4);
  UNUSED_RELEASE(bias_shape);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data)
  {
    assert(bias_shape.FlatSize() == output_depth);
  }
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  for (int batch = 0; batch < batches; ++batch)
  {
    for (int out_y = 0; out_y < output_height; ++out_y)
    {
      for (int out_x = 0; out_x < output_width; ++out_x)
      {
        for (int out_channel = 0; out_channel < output_depth; ++out_channel)
        {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          const int in_y_origin = (out_y * stride_height) - pad_height;
          int32_t acc = 0;
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
                const int in_base = Offset(input_shape, batch, in_y, in_x, 0);
                const int filter_base = Offset(filter_shape, out_channel, filter_y, filter_x, 0);
                for (int in_channel = 0; in_channel < input_depth; in_channel++)
                {
                  int32_t input_val = input_data[in_channel + in_base];
                  int32_t filter_val = filter_data[in_channel + filter_base];
                  acc += (filter_val + filter_offset) * (input_val + input_offset);
                }
              }
            }
          }
          if (bias_data)
          {
            acc += bias_data[out_channel];
          }
          acc = MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
          acc += output_offset;
          acc = std::max(acc, output_activation_min);
          acc = std::min(acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
            static_cast<uint8_t>(acc);
        }
      }
    }
  }
}

template <typename T, bool is_asymmetric>
inline void Conv(const ConvParams &params, const int32_t *output_multiplier,
                 const int32_t *output_shift, const Shape &input_shape, const T *input_data,
                 const Shape &filter_shape, const T *filter_data, const int32_t *filter_zeropoint,
                 const Shape &bias_shape, const int32_t *bias_data, const Shape &output_shape,
                 T *output_data)

{
  UNUSED_RELEASE(bias_shape);
  // Get parameters.
  const int32_t input_offset = params.input_offset; // r = s(q - Z)
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t output_offset = params.output_offset;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  assert(output_activation_min < output_activation_max);
  assert(input_shape.DimensionsCount() == 4);
  assert(filter_shape.DimensionsCount() == 4);
  assert(output_shape.DimensionsCount() == 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data)
  {
    assert(bias_shape.FlatSize() == output_depth);
  }

  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  for (int batch = 0; batch < batches; ++batch)
  {
    for (int out_y = 0; out_y < output_height; ++out_y)
    {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      for (int out_x = 0; out_x < output_width; ++out_x)
      {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel)
        {
          int32_t acc = 0;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y)
          {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            for (int filter_x = 0; filter_x < filter_width; ++filter_x)
            {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;

              // Zero padding by omitting the areas outside the image.
              const bool is_point_inside_image =
                (in_x >= 0) && (in_x < input_width) && (in_y >= 0) && (in_y < input_height);

              if (!is_point_inside_image)
              {
                continue;
              }

              for (int in_channel = 0; in_channel < input_depth; ++in_channel)
              {
                const T input_val = input_data[Offset(input_shape, batch, in_y, in_x, in_channel)];
                const T filter_val =
                  filter_data[Offset(filter_shape, out_channel, filter_y, filter_x, in_channel)];
                if (is_asymmetric)
                {
                  const int32_t filter_offset = -filter_zeropoint[out_channel];
                  acc += (filter_val + filter_offset) * (input_val + input_offset);
                }
                else
                {
                  // Accumulate with 32 bits accumulator.
                  // In the nudging process during model quantization, we force
                  // real value of 0.0 be represented by a quantized value. This
                  // guarantees that the input_offset is a int8_t, even though
                  // it is represented using int32_t. int32_t += int8_t *
                  // (int8_t - int8_t) so the highest value we can get from each
                  // accumulation is [-127, 127] * ([-128, 127] -
                  // [-128, 127]), which is [-32512, 32512]. log2(32512)
                  // = 14.98, which means we can accumulate at least 2^16
                  // multiplications without overflow. The accumulator is
                  // applied to a filter so the accumulation logic will hold as
                  // long as the filter size (filter_y * filter_x * in_channel)
                  // does not exceed 2^16, which is the case in all the models
                  // we have seen so far.
                  // TODO(jianlijianli): Add a check to make sure the
                  // accumulator depth is smaller than 2^16.
                  acc += filter_val * (input_val + input_offset);
                  UNUSED_RELEASE(filter_zeropoint);
                }
              }
            }
          }

          if (bias_data)
          {
            acc += bias_data[out_channel];
          }
          acc = MultiplyByQuantizedMultiplier(acc, output_multiplier[out_channel],
                                              output_shift[out_channel]);
          acc += output_offset;
          acc = std::max(acc, output_activation_min);
          acc = std::min(acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] = static_cast<T>(acc);
        }
      }
    }
  }
}

// Copied from tflite 2.13.0
inline void Conv(const ConvParams &params, float *scaling_factors_ptr, const Shape &input_shape,
                 const int8_t *input_data, const Shape &filter_shape, const int8_t *filter_data,
                 const Shape &bias_shape, const float *bias_data, const Shape &output_shape,
                 float *output_data, const float *per_channel_scale, const int32_t *input_offset)

{
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const float output_activation_min = params.float_activation_min;
  const float output_activation_max = params.float_activation_max;
  assert(input_shape.DimensionsCount() == 4);
  assert(filter_shape.DimensionsCount() == 4);
  assert(output_shape.DimensionsCount() == 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data)
  {
    assert(bias_shape.FlatSize() == output_depth);
  }
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  const int groups = input_depth / filter_input_depth;
  assert(input_depth % filter_input_depth == 0);
  const int filters_per_group = output_depth / groups;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  for (int batch = 0; batch < batches; ++batch)
  {
    for (int out_y = 0; out_y < output_height; ++out_y)
    {
      for (int out_x = 0; out_x < output_width; ++out_x)
      {
        for (int out_channel = 0; out_channel < output_depth; ++out_channel)
        {
          auto group = out_channel / filters_per_group;
          const int in_x_origin = (out_x * stride_width) - pad_width;
          const int in_y_origin = (out_y * stride_height) - pad_height;
          int32_t acc = 0;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y)
          {
            for (int filter_x = 0; filter_x < filter_width; ++filter_x)
            {
              for (int in_channel = 0; in_channel < filter_input_depth; ++in_channel)
              {
                const int in_x = in_x_origin + dilation_width_factor * filter_x;
                const int in_y = in_y_origin + dilation_height_factor * filter_y;
                // If the location is outside the bounds of the input image,
                // use zero as a default value.
                if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) && (in_y < input_height))
                {
                  int32_t input_val = input_data[Offset(input_shape, batch, in_y, in_x,
                                                        in_channel + group * filter_input_depth)];
                  int32_t filter_val =
                    filter_data[Offset(filter_shape, out_channel, filter_y, filter_x, in_channel)];
                  acc += filter_val * (input_val - input_offset[batch]);
                }
              }
            }
          }
          float acc_float = acc * per_channel_scale[out_channel] * scaling_factors_ptr[batch];
          if (bias_data)
          {
            acc_float += bias_data[out_channel];
          }
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
            ActivationFunctionWithMinMax(acc_float, output_activation_min, output_activation_max);
        }
      }
    }
  }
}

} // namespace reference
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_REFERENCE_CONV_H__
