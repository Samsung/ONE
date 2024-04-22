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

#include "PALFullyConnectedCommon.h"

namespace luci_interpreter_pal
{

//template <>
//inline void FullyConnected(const luci_interpreter_pal::FullyConnectedParams &params,
//                           const int32_t *input_shape, const int8_t *input_data,
//                           const int32_t *filter_shape, const int8_t *filter_data,
//                           const int32_t *bias_data, const int32_t *output_shape,
//                           int8_t *output_data, uint32_t output_dims_count, uint32_t weights_dims_count)
//{
//  // MARK: At this moment this operation doesn't support
//  const int32_t input_offset = params.input_offset;
//  const int32_t filter_offset = params.filter_offset;
//  const int32_t output_offset = params.output_offset;
//  const int32_t output_multiplier = params.output_multiplier;
//  const int output_shift = params.output_shift;
//  const int32_t output_activation_min = params.quantized_activation_min;
//  const int32_t output_activation_max = params.quantized_activation_max;
// // TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
// // TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);
//
// // TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
//  //const int filter_dim_count = filter_shape.DimensionsCount();
//  //const int output_dim_count = output_shape.DimensionsCount();
//
//  const int batches = flatSizeSkipDim(output_shape, output_dims_count - 1, output_dims_count);
//  const int output_depth = output_shape[output_dims_count - 1];
// // TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2));
//  //const int accum_depth = filter_shape.Dims(filter_dim_count - 1);
//  const int accum_depth = filter_shape[weights_dims_count - 1];
//  for (int b = 0; b < batches; ++b) {
//    for (int out_c = 0; out_c < output_depth; ++out_c) {
//      int32_t acc = 0;
//      for (int d = 0; d < accum_depth; ++d) {
//        int32_t input_val = input_data[b * accum_depth + d];
//        int32_t filter_val = filter_data[out_c * accum_depth + d];
//        acc += (filter_val + filter_offset) * (input_val + input_offset);
//      }
//      if (bias_data) {
//        acc += bias_data[out_c];
//      }
//      int32_t acc_scaled =
//        MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
//      acc_scaled += output_offset;
//      acc_scaled = std::max(acc_scaled, output_activation_min);
//      acc_scaled = std::min(acc_scaled, output_activation_max);
//      output_data[out_c + output_depth * b] =
//        static_cast<OutputType>(acc_scaled);
//    }
//  }
//}

//template <typename InputType, typename WeightType, typename OutputType,
//  typename BiasType>
//void FullyConnected(const FullyConnectedParams& params,
//                    const RuntimeShape& input_shape,
//                    const InputType* input_data,
//                    const RuntimeShape& filter_shape,
//                    const WeightType* filter_data,
//                    const RuntimeShape& bias_shape, const BiasType* bias_data,
//                    const RuntimeShape& output_shape, OutputType* output_data) {
//
//}

template <>
inline void FullyConnected(const luci_interpreter_pal::FullyConnectedParams &, const int32_t *,
                           const int16_t *, const int32_t *, const int8_t *, const int64_t *,
                           const int32_t *, int16_t *, uint32_t, uint32_t)
{
  // MARK: At this moment this operation doesn't support
  assert(false && "FullyConnected INT16 NYI");
}

} // namespace luci_interpreter_pal

#endif // LUCI_INTERPRETER_PAL_FULLY_CONNECTED_H
