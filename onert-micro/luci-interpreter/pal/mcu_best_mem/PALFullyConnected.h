/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef LUCI_INTERPRETER_PAL_FULLY_CONNECTED_H
#define LUCI_INTERPRETER_PAL_FULLY_CONNECTED_H

#include "Params.h"
#include "PALUtils.h"

namespace luci_interpreter_pal
{
template <typename InputType, typename WeightType, typename OutputType, typename BiasType>
inline void FullyConnected(const FullyConnectedParams &params, const int32_t *input_shape,
                           const InputType *input_data, const int32_t *filter_shape,
                           const WeightType *filter_data, const BiasType *bias_data,
                           const int32_t *output_shape, OutputType *output_data,
                           luci_interpreter::OperationGraphStatus)
{
  const int32_t input_offset = params.input_offset;
  const int32_t filter_offset = params.weights_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  const int batches = input_shape[0];
  const int output_depth = output_shape[1];
  const int accum_depth = filter_shape[1];

  for (int b = 0; b < batches; ++b)
  {
    for (int out_c = 0; out_c < output_depth; ++out_c)
    {
      BiasType acc = 0;
      for (int d = 0; d < accum_depth; ++d)
      {
        int32_t input_val = input_data[b * accum_depth + d];
        int32_t filter_val = filter_data[out_c * accum_depth + d];
        acc += (filter_val + filter_offset) * (input_val + input_offset);
      }
      if (bias_data)
      {
        acc += bias_data[out_c];
      }
      int32_t acc_scaled = multiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
      acc_scaled += output_offset;
      acc_scaled = std::max(acc_scaled, output_activation_min);
      acc_scaled = std::min(acc_scaled, output_activation_max);
      output_data[out_c + output_depth * b] = static_cast<OutputType>(acc_scaled);
    }
  }
}
template <>
inline void FullyConnected(const FullyConnectedParams &params, const int32_t *input_shape,
                           const float *input_data, const int32_t *filter_shape,
                           const float *filter_data, const float *bias_data,
                           const int32_t *output_shape, float *output_data,
                           luci_interpreter::OperationGraphStatus status)
{
  const float output_activation_min = params.float_activation_min;
  const float output_activation_max = params.float_activation_max;

  const int batches = input_shape[0];
  const int output_depth = output_shape[1];
  const int accum_depth = filter_shape[1];

  // Calculate scales
  constexpr float max_int16_value = 32767.0f;
  // Input scale
  const float input_scale = params.input_min_max_range / max_int16_value;
  // Output scale
  const float output_scale = params.output_min_max_range / max_int16_value;

  // Create int16 input and output pointers for not USUAL status
  const int16_t *int16_input_data = reinterpret_cast<const int16_t *>(input_data);
  int16_t *int16_output_data = reinterpret_cast<int16_t *>(output_data);

  // For Weight Quantize
  const int8_t *filter_int8_data = reinterpret_cast<const int8_t *>(filter_data);

  float filter_scale = 0.f;
  if (params.is_weight_quant)
    filter_scale = params.weight_scale;

  for (int b = 0; b < batches; ++b)
  {
    for (int out_c = 0; out_c < output_depth; ++out_c)
    {
      float total = 0.f;
      for (int d = 0; d < accum_depth; ++d)
      {
        float input_value = 0.f;
        if (status == luci_interpreter::OperationGraphStatus::MIDDLE or
            status == luci_interpreter::OperationGraphStatus::END)
          input_value = static_cast<float>(int16_input_data[b * accum_depth + d]) * input_scale;
        else
          input_value = input_data[b * accum_depth + d];

        float filter_value = 0.f;
        if (params.is_weight_quant == false)
          filter_value = filter_data[out_c * accum_depth + d];
        else
          filter_value =
            static_cast<float>(filter_int8_data[out_c * accum_depth + d]) * filter_scale;

        total += input_value * filter_value;
      }

      float bias_value = 0.0f;
      if (bias_data)
      {
        bias_value = bias_data[out_c];
      }
      if (status == luci_interpreter::OperationGraphStatus::MIDDLE or
          status == luci_interpreter::OperationGraphStatus::START)
      {
        total =
          std::min(std::max(total + bias_value, output_activation_min), output_activation_max);

        const int16_t result = static_cast<int16_t>(std::round(total / output_scale));
        int16_output_data[out_c + output_depth * b] = result;
      }
      else
      {
        output_data[out_c + output_depth * b] =
          std::min(std::max(total + bias_value, output_activation_min), output_activation_max);
      }
    }
  }
}

} // namespace luci_interpreter_pal

#endif // LUCI_INTERPRETER_PAL_FULLY_CONNECTED_H
