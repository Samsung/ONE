/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

#ifndef LUCI_INTERPRETER_PAL_MAX_POOL_2D_H
#define LUCI_INTERPRETER_PAL_MAX_POOL_2D_H

#include "Params.h"
#include "PALUtils.h"

namespace luci_interpreter_pal
{
template <typename T>
inline void MaxPool(const PoolParams &params, const luci_interpreter::RuntimeShape &input_shape,
                    const T *input_data, const luci_interpreter::RuntimeShape &output_shape,
                    T *output_data)
{
  const int batches = input_shape.dims(0);
  const int depth = output_shape.dims(3);
  const int input_height = input_shape.dims(1);
  const int input_width = input_shape.dims(2);
  const int output_height = output_shape.dims(1);
  const int output_width = output_shape.dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;
  for (int batch = 0; batch < batches; ++batch)
  {
    for (int out_y = 0; out_y < output_height; ++out_y)
    {
      for (int out_x = 0; out_x < output_width; ++out_x)
      {
        for (int channel = 0; channel < depth; ++channel)
        {
          const int in_x_origin = (out_x * stride_width) - params.padding_values.width;
          const int in_y_origin = (out_y * stride_height) - params.padding_values.height;
          // Compute the boundaries of the filter region clamped so as to
          // ensure that the filter window fits in the input array.
          const int filter_x_start = std::max(0, -in_x_origin);
          const int filter_x_end = std::min(params.filter_width, input_width - in_x_origin);
          const int filter_y_start = std::max(0, -in_y_origin);
          const int filter_y_end = std::min(params.filter_height, input_height - in_y_origin);
          T max = std::numeric_limits<T>::lowest();
          for (int filter_y = filter_y_start; filter_y < filter_y_end; ++filter_y)
          {
            for (int filter_x = filter_x_start; filter_x < filter_x_end; ++filter_x)
            {
              const int in_x = in_x_origin + filter_x;
              const int in_y = in_y_origin + filter_y;

              const int input_data_offset =
                ((batch * input_shape.dims(1) + in_y) * input_shape.dims(2) + in_x) *
                  input_shape.dims(3) +
                channel;

              max = std::max(max, input_data[input_data_offset]);
            }
          }
          max = std::max<T>(max, params.quantized_activation_min);
          max = std::min<T>(max, params.quantized_activation_max);

          const int output_data_offset =
            ((batch * output_shape.dims(1) + out_y) * output_shape.dims(2) + out_x) *
              output_shape.dims(3) +
            channel;

          output_data[output_data_offset] = static_cast<T>(max);
        }
      }
    }
  }
}

inline void MaxPool(const PoolParams &params, const luci_interpreter::RuntimeShape &input_shape,
                    const float *input_data, const luci_interpreter::RuntimeShape &output_shape,
                    float *output_data, luci_interpreter::OperationGraphStatus status)
{
  const int batches = input_shape.dims(0);
  const int depth = output_shape.dims(3);
  const int input_height = input_shape.dims(1);
  const int input_width = input_shape.dims(2);
  const int output_height = output_shape.dims(1);
  const int output_width = output_shape.dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;

  // Calculate scales
  const float max_int16_value = 32767.0f;
  // Input scale
  const float input_min_max_range = params.input_min_max_range;
  const float input_scale = input_min_max_range / max_int16_value;
  // Output scale
  const float output_min_max_range = params.output_min_max_range;
  const float output_scale = output_min_max_range / max_int16_value;

  // Create int16 input and output pointers for not ORDINAL status
  const int16_t *int16_input_data = reinterpret_cast<const int16_t *>(input_data);
  int16_t *int16_output_data = reinterpret_cast<int16_t *>(output_data);

  for (int batch = 0; batch < batches; ++batch)
  {
    for (int out_y = 0; out_y < output_height; ++out_y)
    {
      for (int out_x = 0; out_x < output_width; ++out_x)
      {
        for (int channel = 0; channel < depth; ++channel)
        {
          const int in_x_origin = (out_x * stride_width) - params.padding_values.width;
          const int in_y_origin = (out_y * stride_height) - params.padding_values.height;
          // Compute the boundaries of the filter region clamped so as to
          // ensure that the filter window fits in the input array.
          const int filter_x_start = std::max(0, -in_x_origin);
          const int filter_x_end = std::min(params.filter_width, input_width - in_x_origin);
          const int filter_y_start = std::max(0, -in_y_origin);
          const int filter_y_end = std::min(params.filter_height, input_height - in_y_origin);
          float max = std::numeric_limits<float>::lowest();
          for (int filter_y = filter_y_start; filter_y < filter_y_end; ++filter_y)
          {
            for (int filter_x = filter_x_start; filter_x < filter_x_end; ++filter_x)
            {
              const int in_x = in_x_origin + filter_x;
              const int in_y = in_y_origin + filter_y;

              const int input_data_offset =
                ((batch * input_shape.dims(1) + in_y) * input_shape.dims(2) + in_x) *
                  input_shape.dims(3) +
                channel;

              float input_value = 0.f;
              if (status == luci_interpreter::OperationGraphStatus::USUAL or
                  status == luci_interpreter::OperationGraphStatus::START)
                input_value = input_data[input_data_offset];
              else
                input_value = static_cast<float>(int16_input_data[input_data_offset]) * input_scale;
              max = std::max(max, input_value);
            }
          }
          const int output_data_offset =
            ((batch * output_shape.dims(1) + out_y) * output_shape.dims(2) + out_x) *
              output_shape.dims(3) +
            channel;

          const float total =
            std::min(std::max(max, params.float_activation_min), params.float_activation_max);
          if (status == luci_interpreter::OperationGraphStatus::USUAL or
              status == luci_interpreter::OperationGraphStatus::END)
          {
            output_data[output_data_offset] = total;
          }
          else
          {
            const int16_t result = static_cast<int16_t>(std::round(total / output_scale));
            int16_output_data[output_data_offset] = result;
          }
        }
      }
    }
  }
}

} // namespace luci_interpreter_pal

#endif // LUCI_INTERPRETER_PAL_MAX_POOL_2D_H
