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

#ifndef __NNFW_CKER_RESIZEBILINEAR_H__
#define __NNFW_CKER_RESIZEBILINEAR_H__

#include "cker/Shape.h"
#include "cker/Types.h"
#include <cmath>

namespace nnfw
{
namespace cker
{

inline void ResizeBilinearKernel2x2(int32_t x0, int32_t x1, int32_t y0, int32_t y1, int32_t x,
                                    int32_t y, int32_t depth, int32_t batch,
                                    const Shape &input_shape, const float *input_data,
                                    const Shape &output_shape, float *output_data)
{
  const int32_t input_width = input_shape.Dims(2);
  const int32_t output_width = output_shape.Dims(2);

  const int32_t input_x_offset = (x1 - x0) * depth;
  const int32_t input_y_offset = (y1 - y0) * depth * input_width;
  const int32_t output_x_offset = depth;
  const int32_t output_y_offset = depth * output_width;

  for (int ch = 0; ch < depth; ch++)
  {
    const int32_t input_offset = Offset(input_shape, batch, y0, x0, ch);

    float x0y0 = input_data[input_offset];
    float x1y0 = input_data[input_offset + input_x_offset];
    float x0y1 = input_data[input_offset + input_y_offset];
    float x1y1 = input_data[input_offset + input_x_offset + input_y_offset];

    // Top left corner.
    const int32_t output_offset = Offset(output_shape, batch, y, x, ch);
    output_data[output_offset] = x0y0;

    // Top right corner.
    output_data[output_offset + output_x_offset] = (x0y0 + x1y0) / 2;

    // Bottom left corner.
    float output = (x0y0 + x0y1) / 2;
    output_data[output_offset + output_y_offset] = output;

    // Bottom right corner.
    output_data[output_offset + output_x_offset + output_y_offset] =
      (output + ((x1y0 + x1y1) / 2)) / 2;
  }
}

inline void ResizeBilinear2x2(int32_t batches, int32_t input_height, int32_t input_width,
                              int32_t depth, int32_t output_height, int32_t output_width,
                              const Shape &input_shape, const float *input_data,
                              const Shape &output_shape, float *output_data)
{
  for (int b = 0; b < batches; b++)
  {
    for (int y0 = 0, y = 0; y <= output_height - 2; y += 2, y0++)
    {
      for (int x0 = 0, x = 0; x <= output_width - 2; x += 2, x0++)
      {
        int32_t x1 = std::min(x0 + 1, input_width - 1);
        int32_t y1 = std::min(y0 + 1, input_height - 1);
        ResizeBilinearKernel2x2(x0, x1, y0, y1, x, y, depth, b, input_shape, input_data,
                                output_shape, output_data);
      }
    }
  }
}

inline void ResizeBilinearKernel(const float *input_ptr, int32_t depth, float scale,
                                 float *output_ptr)
{
  for (int32_t i = 0; i < depth; i++)
  {
    *output_ptr += *input_ptr * scale;
    output_ptr++;
    input_ptr++;
  }
}

inline void ComputeInterpolationValues(const float value, const float scale,
                                       const bool half_pixel_centers, int32_t input_size,
                                       float *scaled_value, int32_t *lower_bound,
                                       int32_t *upper_bound)
{
  if (half_pixel_centers)
  {
    *scaled_value = (value + 0.5f) * scale - 0.5f;
  }
  else
  {
    *scaled_value = value * scale;
  }
  float scaled_value_floor = std::floor(*scaled_value);
  *lower_bound = std::max(static_cast<int32_t>(scaled_value_floor), static_cast<int32_t>(0));
  *upper_bound = std::min(static_cast<int32_t>(std::ceil(*scaled_value)), input_size - 1);
}

inline void ResizeBilinearGeneric(int32_t batches, int32_t input_height, int32_t input_width,
                                  int32_t depth, int32_t output_height, int32_t output_width,
                                  float height_scale, float width_scale, const Shape &input_shape,
                                  const float *input_data, float *output_data,
                                  const bool half_pixel_centers)
{
  memset(output_data, 0, batches * output_height * output_width * depth * sizeof(float));

  int32_t output_offset = 0;
  for (int b = 0; b < batches; ++b)
  {
    for (int y = 0; y < output_height; ++y)
    {
      float input_y;
      int32_t y0, y1;
      ComputeInterpolationValues(y, height_scale, half_pixel_centers, input_height, &input_y, &y0,
                                 &y1);
      for (int x = 0; x < output_width; ++x)
      {
        float input_x;
        int32_t x0, x1;
        ComputeInterpolationValues(x, width_scale, half_pixel_centers, input_width, &input_x, &x0,
                                   &x1);
        float *output_ptr = &output_data[output_offset];

        // Run kernel on the 4 corners of the bilinear resize algorithm.
        int32_t input_offset = Offset(input_shape, b, y0, x0, 0);
        float scale = (1 - (input_y - y0)) * (1 - (input_x - x0));
        const float *input_ptr = &input_data[input_offset];
        ResizeBilinearKernel(input_ptr, depth, scale, output_ptr);

        input_offset = Offset(input_shape, b, y0, x1, 0);
        scale = (1 - (input_y - y0)) * (input_x - x0);
        input_ptr = &input_data[input_offset];
        ResizeBilinearKernel(input_ptr, depth, scale, output_ptr);

        input_offset = Offset(input_shape, b, y1, x0, 0);
        scale = (input_y - y0) * (1 - (input_x - x0));
        input_ptr = &input_data[input_offset];
        ResizeBilinearKernel(input_ptr, depth, scale, output_ptr);

        input_offset = Offset(input_shape, b, y1, x1, 0);
        scale = (input_y - y0) * (input_x - x0);
        input_ptr = &input_data[input_offset];
        ResizeBilinearKernel(input_ptr, depth, scale, output_ptr);

        output_offset += depth;
      }
    }
  }
}

template <typename T>
inline void ResizeBilinearGenericSmallChannel(int32_t batches, int32_t input_height,
                                              int32_t input_width, int32_t depth,
                                              int32_t output_height, int32_t output_width,
                                              float height_scale, float width_scale,
                                              const Shape &input_shape, const T *input_data,
                                              T *output_data, const bool half_pixel_centers)
{
  T *output_ptr = &output_data[0];
  for (int b = 0; b < batches; ++b)
  {
    for (int y = 0; y < output_height; ++y)
    {
      float input_y;
      int32_t y0, y1;
      ComputeInterpolationValues(y, height_scale, half_pixel_centers, input_height, &input_y, &y0,
                                 &y1);
      for (int x = 0; x < output_width; ++x)
      {
        float input_x;
        int32_t x0, x1;
        ComputeInterpolationValues(x, width_scale, half_pixel_centers, input_width, &input_x, &x0,
                                   &x1);

        int32_t input_offset[4] = {
          Offset(input_shape, b, y0, x0, 0), Offset(input_shape, b, y0, x1, 0),
          Offset(input_shape, b, y1, x0, 0), Offset(input_shape, b, y1, x1, 0)};
        float scale[4] = {(1 - (input_y - y0)) * (1 - (input_x - x0)),
                          (1 - (input_y - y0)) * (input_x - x0),
                          (input_y - y0) * (1 - (input_x - x0)), (input_y - y0) * (input_x - x0)};

        for (int d = 0; d < depth; d++)
        {
          const T *input_ptr = &input_data[d];
          *output_ptr++ = static_cast<T>(
            input_ptr[input_offset[0]] * scale[0] + input_ptr[input_offset[1]] * scale[1] +
            input_ptr[input_offset[2]] * scale[2] + input_ptr[input_offset[3]] * scale[3]);
        }
      }
    }
  }
}

void ResizeBilinear(ResizeBilinearParams &params, const Shape &input_shape, const float *input_data,
                    const Shape &output_shape, float *output_data)
{
  int32_t batches = static_cast<int32_t>(MatchingDim(input_shape, 0, output_shape, 0));
  int32_t input_height = input_shape.Dims(1);
  int32_t input_width = input_shape.Dims(2);
  int32_t depth = static_cast<int32_t>(MatchingDim(input_shape, 3, output_shape, 3));

  // Specialize for 2x2 upsample.
  if (!params.align_corners && !params.half_pixel_centers &&
      params.output_height == 2 * input_height && params.output_width == 2 * input_width)
  {
    ResizeBilinear2x2(batches, input_height, input_width, depth, params.output_height,
                      params.output_width, input_shape, input_data, output_shape, output_data);
  }
  else
  {
    float height_scale = static_cast<float>(input_height) / params.output_height;
    float width_scale = static_cast<float>(input_width) / params.output_width;
    if (params.align_corners && params.output_height > 1)
    {
      height_scale = static_cast<float>(input_height - 1) / (params.output_height - 1);
    }
    if (params.align_corners && params.output_width > 1)
    {
      width_scale = static_cast<float>(input_width - 1) / (params.output_width - 1);
    }

    ResizeBilinearGeneric(batches, input_height, input_width, depth, params.output_height,
                          params.output_width, height_scale, width_scale, input_shape, input_data,
                          output_data, params.half_pixel_centers);
  }
}

void ResizeBilinear(ResizeBilinearParams &params, const Shape &input_shape,
                    const uint8_t *input_data, const Shape &output_shape, uint8_t *output_data)
{
  int32_t batches = MatchingDim(input_shape, 0, output_shape, 0);
  int32_t input_height = input_shape.Dims(1);
  int32_t input_width = input_shape.Dims(2);
  int32_t depth = MatchingDim(input_shape, 3, output_shape, 3);

  float height_scale = (params.align_corners && params.output_height > 1)
                         ? (static_cast<float>(input_height - 1) / (params.output_height - 1))
                         : (static_cast<float>(input_height) / params.output_height);

  float width_scale = (params.align_corners && params.output_width > 1)
                        ? (static_cast<float>(input_width - 1) / (params.output_width - 1))
                        : (static_cast<float>(input_width) / params.output_width);

  ResizeBilinearGenericSmallChannel<uint8_t>(
    batches, input_height, input_width, depth, params.output_height, params.output_width,
    height_scale, width_scale, input_shape, input_data, output_data, params.half_pixel_centers);
}
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_RESIZEBILINEAR_H__
