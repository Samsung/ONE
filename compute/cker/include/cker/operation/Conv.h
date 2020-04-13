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

#ifndef __NNFW_CKER_CONV_H__
#define __NNFW_CKER_CONV_H__

#include "cker/Types.h"
#include "cker/Shape.h"
#include "cker/Utils.h"
#include "cker/operation/reference/Conv.h"
#include "cker/operation/optimized/Conv.h"
#include <vector>

namespace nnfw
{
namespace cker
{

namespace
{
// Naive implementation of transpose for floats. Could be optimized to be more
// cache friendly, but for now it's a one-time cost on first run, and we would
// prefer to remove the need to do this at all eventually.
inline void TransposeFloatTensor(const float *input_data, const nnfw::cker::Shape &output_shape,
                                 float *output_data)
{
  const int rows = output_shape.Dims(1);
  const int cols = output_shape.Dims(0);
  for (int i = 0; i < rows; ++i)
  {
    for (int j = 0; j < cols; ++j)
    {
      const float in_value = input_data[i * cols + j];
      output_data[j * rows + i] = in_value;
    }
  }
}
} // namespace

class Conv
{
public:
  Conv() : _modified_filter_data(), _prepared(false) {}

  void prepare(const Shape &filter_shape, const float *filter_data, PaddingType padding_type,
               bool &is_replaced_weights)
  {
    (void)filter_shape;
    (void)filter_data;
    (void)padding_type;
    (void)is_replaced_weights;
    if (!_prepared)
    {
      if (padding_type != PaddingType::kNone && std::thread::hardware_concurrency() > 1)
      {
        const auto output_depth = filter_shape.Dims(0);
        const Shape hwcn_filter_shape{filter_shape.FlatSize() / output_depth, output_depth};
        _modified_filter_data.resize(hwcn_filter_shape.FlatSize());
        TransposeFloatTensor(filter_data, hwcn_filter_shape, &_modified_filter_data[0]);
        is_replaced_weights = true;
      }
      _prepared = true;
    }
  }

  void operator()(const ConvParams &params, const Shape &input_shape, const float *input_data,
                  const Shape &filter_shape, const float *filter_data, const Shape &bias_shape,
                  const float *bias_data, const Shape &output_shape, float *output_data)
  {
    if (params.padding_type != PaddingType::kNone && std::thread::hardware_concurrency() > 1)
    {
      if (!_prepared)
      {
        bool not_used_condition = false;
        prepare(filter_shape, filter_data, params.padding_type, not_used_condition);
        _prepared = true;
      }
      multithreaded::Conv(params, input_shape, input_data, filter_shape, &_modified_filter_data[0],
                          bias_shape, bias_data, output_shape, output_data);
    }
    else
    {
      // TODO Support optimized kernel
      reference::Conv(params, input_shape, input_data, filter_shape, filter_data, bias_shape,
                      bias_data, output_shape, output_data);
    }
  }

  void operator()(const ConvParams &params, const Shape &input_shape, const uint8_t *input_data,
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
                const int in_base = Offset(input_shape, batch, in_y, in_x, 0);
                const int filter_base = Offset(filter_shape, out_channel, filter_y, filter_x, 0);
                if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) && (in_y < input_height))
                {
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

private:
  std::vector<float> _modified_filter_data;
  bool _prepared;
};
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_CONCATENATION_H_
