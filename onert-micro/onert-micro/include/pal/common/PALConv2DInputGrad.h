/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef ONERT_MICRO_PAL_CONV2D_WEIGHT_GRAD_COMMON_H
#define ONERT_MICRO_PAL_CONV2D_WEIGHT_GRAD_COMMON_H

#include "Params.h"
#include "PALUtils.h"

#include "OMStatus.h"

namespace onert_micro
{
namespace execute
{
namespace pal
{

namespace
{

/*
  * Rotate square 2D weights by 180 degrees
  */

void rotate_180(float *weights, int numRows, int numCols)
{
  // Rotate cols
  for (int row = 0; row < numRows; ++row)
  {
    for (int col = 0; col < numCols / 2; ++col)
    {
      float tmp_value = weights[row * numCols + col];
      weights[row * numCols + col] = weights[row * numCols + numCols - col - 1];
      weights[row * numCols + numCols - col - 1] = tmp_value;
    }
  }

  // todo: add rotate rows
}

} // namespace

OMStatus ConvInputGradFloat(const core::FloatConv2D *params, const core::OMRuntimeShape &input_shape,
                             const float *input_data, const core::OMRuntimeShape &weight_shape,
                             const float *weight_data,
                             const core::OMRuntimeShape &output_shape, float *output_data)
{
  const int stride_width = params->stride_w;
  const int stride_height = params->stride_h;
  const int dilation_width_factor = params->dilation_width_factor;
  const int dilation_height_factor = params->dilation_height_factor;
  const int pad_width = params->pad_w;
  const int pad_height = params->pad_h;

  const int batches = input_shape.dims(0);
  const int input_height = input_shape.dims(2);
  const int input_width = input_shape.dims(3);
  const int output_depth = input_shape.dims(1);
  const int input_depth = weight_shape.dims(1);
  const int filter_height = weight_shape.dims(2);
  const int filter_width = weight_shape.dims(3);
  const int output_height = output_shape.dims(2);
  const int output_width = output_shape.dims(3);

  auto n_c_weight_data = const_cast<float *>(weight_data);

  for (int b = 0; b < batches; ++b)
  {
    for (int oc = 0; oc < output_depth; ++oc)
    {
      for (int ic = 0; ic < input_depth; ++ic)
      {
        // rotate;
        rotate_180(n_c_weight_data + offset(weight_shape.dimsData(), oc, ic,
                          0, 0), filter_height, filter_width);
        for (int out_y = 0; out_y < output_height; ++out_y)
        {
          for (int out_x = 0; out_x < output_width; ++out_x)
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
                  float input_value = input_data[offset(
                    input_shape.dimsData(), b, oc, in_y, in_x)];
                  float filter_value = weight_data[offset(weight_shape.dimsData(), oc, ic,
                                                              filter_y, filter_x)];
                  total += (input_value * filter_value);
                }
              }
            }
            auto tmp = offset(output_shape.dimsData(), b, ic, out_y, out_x);
            output_data[offset(output_shape.dimsData(), b, ic, out_y, out_x)] = total;
          }
        }
        // rotate back
        rotate_180(n_c_weight_data + offset(weight_shape.dimsData(), oc, ic,
                          0, 0), filter_height, filter_width);
      }
    }
  }
  return Ok;
}

} // namespace pal
} // namespace execute
} // namespace onert_micro

#endif // ONERT_MICRO_PAL_CONV2D_WEIGHT_GRAD_COMMON_H
