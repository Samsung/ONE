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
OMStatus ConvWeightGradFloat(const core::FloatConv2D *params, const core::OMRuntimeShape &input_activation_shape,
                   const float *input_activation_data, const core::OMRuntimeShape &input_grad_shape,
                   const float *input_grad_data,
                   const core::OMRuntimeShape &output_shape, float *output_data)
{
  const int stride_width = params->stride_w;
  const int stride_height = params->stride_h;
  const int dilation_width_factor = params->dilation_width_factor;
  const int dilation_height_factor = params->dilation_height_factor;
  const int pad_width = params->pad_w;
  const int pad_height = params->pad_h;

  const int batches = input_grad_shape.dims(0);
  const int input_height = input_activation_shape.dims(2);
  const int input_width = input_activation_shape.dims(3);
  const int input_depth = input_activation_shape.dims(1);
  const int output_depth = input_grad_shape.dims(1);
  const int filter_height = input_grad_shape.dims(2);
  const int filter_width = input_grad_shape.dims(3);
  const int output_height = output_shape.dims(2);
  const int output_width = output_shape.dims(3);

  for (int b = 0; b < batches; ++b)
  {
      for (int oc = 0; oc < output_depth; ++oc)
      {
        for (int ic = 0; ic < input_depth; ++ic)
        {
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
                    float input_value = input_activation_data[offset(
                      input_activation_shape.dimsData(), b, ic, in_y, in_x)];
                    float filter_value = input_grad_data[offset(input_grad_shape.dimsData(), b, oc,
                                                                filter_y, filter_x)];
                    total += (input_value * filter_value);
                  }
                }
              }
              auto tmp = offset(output_shape.dimsData(), oc, ic, out_y, out_x);
              output_data[offset(output_shape.dimsData(), oc, ic, out_y, out_x)] = total;
            }
          }
        }
    }
  }
  return Ok;
}

} // namespace pal
} // namespace execute
} // namespace onert_micro

#endif // ONERT_MICRO_PAL_CONV2D_WEIGHT_GRAD_COMMON_H
