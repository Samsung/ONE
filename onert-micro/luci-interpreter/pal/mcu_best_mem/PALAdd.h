/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef LUCI_INTERPRETER_PAL_ADD_H
#define LUCI_INTERPRETER_PAL_ADD_H

#include "Params.h"
#include "PALUtils.h"
#include "ProcessBroadcastShapes.h"

namespace luci_interpreter_pal
{

template <typename T>
inline void Add(const ArithmeticParams &params, const int flat_size, const T *input1_data,
                const T *input2_data, T *output_data, luci_interpreter::OperationGraphStatus)
{
  T activation_min, activation_max;
  getActivationParams(params, &activation_min, &activation_max);

  for (int i = 0; i < flat_size; ++i)
    output_data[i] =
      std::min(std::max(input1_data[i] + input2_data[i], activation_min), activation_max);
}

// Quantized intermediate tensors
template <>
inline void Add(const ArithmeticParams &params, const int flat_size, const float *input1_data,
                const float *input2_data, float *output_data,
                luci_interpreter::OperationGraphStatus status)
{
  float activation_min, activation_max;
  getActivationParams(params, &activation_min, &activation_max);

  if (status == luci_interpreter::OperationGraphStatus::USUAL)
  {
    for (int i = 0; i < flat_size; ++i)
      output_data[i] =
        std::min(std::max(input1_data[i] + input2_data[i], activation_min), activation_max);
    return;
  }

  // Calculate scales
  const float max_int16_value = 32767.0f;
  // Input scale
  const float input1_min_max_range = params.input1_min_max_range;
  const float input1_scale = input1_min_max_range / max_int16_value;
  const float input2_min_max_range = params.input2_min_max_range;
  const float input2_scale = input2_min_max_range / max_int16_value;
  // Output scale
  const float output_min_max_range = params.output_min_max_range;
  const float output_scale = output_min_max_range / max_int16_value;

  // Create int16 input and output pointers for not USUAL status
  const int16_t *int16_input_data_1 = reinterpret_cast<const int16_t *>(input1_data);
  const int16_t *int16_input_data_2 = reinterpret_cast<const int16_t *>(input2_data);
  int16_t *int16_output_data = reinterpret_cast<int16_t *>(output_data);

  if (status == luci_interpreter::OperationGraphStatus::START)
  {
    for (int i = 0; i < flat_size; ++i)
    {
      float output_value =
        std::min(std::max(input1_data[i] + input2_data[i], activation_min), activation_max);
      int16_output_data[i] = static_cast<int16_t>(std::round(output_value / output_scale));
    }
  }
  else if (status == luci_interpreter::OperationGraphStatus::END)
  {
    for (int i = 0; i < flat_size; ++i)
    {
      const float input1_value = static_cast<float>(int16_input_data_1[i]) * input1_scale;
      const float input2_value = static_cast<float>(int16_input_data_2[i]) * input2_scale;
      float output_value =
        std::min(std::max(input1_value + input2_value, activation_min), activation_max);
      output_data[i] = output_value;
    }
  }
  else // MIDDLE
  {
    for (int i = 0; i < flat_size; ++i)
    {
      const float input1_value = static_cast<float>(int16_input_data_1[i]) * input1_scale;
      const float input2_value = static_cast<float>(int16_input_data_2[i]) * input2_scale;
      float output_value =
        std::min(std::max(input1_value + input2_value, activation_min), activation_max);
      int16_output_data[i] = static_cast<int16_t>(std::round(output_value / output_scale));
    }
  }
}

template <typename T>
inline void AddScalar(const ArithmeticParams &params, const int flat_size, const T *input_data,
                      const T scalar_value, T *output_data, luci_interpreter::OperationGraphStatus)
{
  T activation_min, activation_max;
  getActivationParams(params, &activation_min, &activation_max);

  for (int i = 0; i < flat_size; ++i)
    output_data[i] =
      std::min(std::max(input_data[i] + scalar_value, activation_min), activation_max);
}

// Quantized intermediate tensors
template <>
inline void AddScalar(const ArithmeticParams &params, const int flat_size, const float *input_data,
                      const float scalar_value, float *output_data,
                      luci_interpreter::OperationGraphStatus status)
{
  float activation_min, activation_max;
  getActivationParams(params, &activation_min, &activation_max);

  if (status == luci_interpreter::OperationGraphStatus::USUAL)
  {
    for (int i = 0; i < flat_size; ++i)
      output_data[i] =
        std::min(std::max(input_data[i] + scalar_value, activation_min), activation_max);
    return;
  }

  // Calculate scales
  const float max_int16_value = 32767.0f;
  // Input scale
  const float input_min_max_range = params.input1_min_max_range;
  const float input_scale = input_min_max_range / max_int16_value;
  // Output scale
  const float output_min_max_range = params.output_min_max_range;
  const float output_scale = output_min_max_range / max_int16_value;

  // Create int16 input and output pointers for not USUAL status
  const int16_t *int16_input_data = reinterpret_cast<const int16_t *>(input_data);
  int16_t *int16_output_data = reinterpret_cast<int16_t *>(output_data);

  if (status == luci_interpreter::OperationGraphStatus::START)
  {
    for (int i = 0; i < flat_size; ++i)
    {
      float output_value =
        std::min(std::max(input_data[i] + scalar_value, activation_min), activation_max);
      int16_output_data[i] = static_cast<int16_t>(std::round(output_value / output_scale));
    }
  }
  else if (status == luci_interpreter::OperationGraphStatus::END)
  {
    for (int i = 0; i < flat_size; ++i)
    {
      const float input_value = static_cast<float>(int16_input_data[i]) * input_scale;
      float output_value =
        std::min(std::max(input_value + scalar_value, activation_min), activation_max);
      output_data[i] = output_value;
    }
  }
  else // MIDDLE
  {
    for (int i = 0; i < flat_size; ++i)
    {
      const float input_value = static_cast<float>(int16_input_data[i]) * input_scale;
      float output_value =
        std::min(std::max(input_value + scalar_value, activation_min), activation_max);
      int16_output_data[i] = static_cast<int16_t>(std::round(output_value / output_scale));
    }
  }
}

template <typename T>
inline void
BroadcastAdd4DSlow(const ArithmeticParams &params,
                   const luci_interpreter::RuntimeShape &input1_shape, const T *input1_data,
                   const luci_interpreter::RuntimeShape &input2_shape, const T *input2_data,
                   const luci_interpreter::RuntimeShape &output_shape, T *output_data,
                   luci_interpreter::OperationGraphStatus status)
{
  const int flat_size = input1_shape.flatSize();

  if (params.broadcast_category == BroadcastableOpCategory::kScalarFirstBroadcast)
  {
    return AddScalar(params, flat_size, input2_data, input1_data[0], output_data, status);
  }
  else if (params.broadcast_category == BroadcastableOpCategory::kScalarSecondBroadcast)
  {
    return AddScalar(params, flat_size, input1_data, input2_data[0], output_data, status);
  }

  // assert(status == luci_interpreter::OperationGraphStatus::USUAL);

  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_shape, input2_shape, &desc1, &desc2);
  const luci_interpreter::RuntimeShape extended_output_shape =
    luci_interpreter::RuntimeShape::extendedShape(4, output_shape);

  T activation_min, activation_max;
  getActivationParams(params, &activation_min, &activation_max);

  const int16_t *input1_data_int16 = reinterpret_cast<const int16_t *>(input1_data);
  const int16_t *input2_data_int16 = reinterpret_cast<const int16_t *>(input2_data);

  int16_t *output_data_int16 = reinterpret_cast<int16_t *>(output_data);

  // Calculate scales
  const float max_int16_value = 32767.0f;
  // Input scale
  const float input1_min_max_range = params.input1_min_max_range;
  const float input1_scale = input1_min_max_range / max_int16_value;
  // Output scale
  const float output_min_max_range = params.output_min_max_range;
  const float output_scale = output_min_max_range / max_int16_value;

  // In Tensorflow, the dimensions are canonically named (batch_number, row,
  // col, channel), with extents (batches, height, width, depth), with the
  // trailing dimension changing most rapidly (channels has the smallest stride,
  // typically 1 element).
  //
  // In generated C code, we store arrays with the dimensions reversed. The
  // first dimension has smallest stride.
  //
  // We name our variables by their Tensorflow convention, but generate C code
  // nesting loops such that the innermost loop has the smallest stride for the
  // best cache behavior.
  for (int b = 0; b < extended_output_shape.dims(0); ++b)
  {
    for (int y = 0; y < extended_output_shape.dims(1); ++y)
    {
      for (int x = 0; x < extended_output_shape.dims(2); ++x)
      {
        for (int c = 0; c < extended_output_shape.dims(3); ++c)
        {
          const int output_data_offset =
            ((b * extended_output_shape.dims(1) + y) * extended_output_shape.dims(2) + x) *
              extended_output_shape.dims(3) +
            c;

          T input1_value;
          T output_value;
          if (status == luci_interpreter::OperationGraphStatus::MIDDLE or
              status == luci_interpreter::OperationGraphStatus::END)
          {
            input1_value =
              static_cast<T>(input1_data_int16[subscriptToIndex(desc1, b, y, x, c)]) * input1_scale;
          }
          else
          {
            input1_value = input1_data[subscriptToIndex(desc1, b, y, x, c)];
          }

          if (status == luci_interpreter::OperationGraphStatus::START or
              status == luci_interpreter::OperationGraphStatus::MIDDLE)
          {
            const auto total =
              std::min(std::max(input1_value + input2_data[subscriptToIndex(desc2, b, y, x, c)],
                                activation_min),
                       activation_max);
            const int16_t result = static_cast<int16_t>(std::round(total / output_scale));
            output_data_int16[output_data_offset] = result;
          }
          else
          {
            output_data[output_data_offset] =
              std::min(std::max(input1_value + input2_data[subscriptToIndex(desc2, b, y, x, c)],
                                activation_min),
                       activation_max);
          }
        }
      }
    }
  }
}

} // namespace luci_interpreter_pal

#endif // LUCI_INTERPRETER_PAL_ADD_H
