/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

#ifndef LUCI_INTERPRETER_PAL_FULLYCONNECTED_H
#define LUCI_INTERPRETER_PAL_FULLYCONNECTED_H

#include <tensorflow/lite/kernels/internal/reference/fully_connected.h>
#include <tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h>

namespace luci_interpreter_pal
{
template <typename T>
static inline void FullyConnected(const tflite::FullyConnectedParams &params,
                                  const tflite::RuntimeShape &input_shape, const T *input_data,
                                  const tflite::RuntimeShape &filter_shape, const T *filter_data,
                                  const tflite::RuntimeShape &bias_shape, const int32_t *bias_data,
                                  const tflite::RuntimeShape &output_shape, T *output_data)
{
  {
    // MARK: At this moment this operation doesn't support
    assert(false && "FullyConnected NYI");
    (void)params;
    (void)input_shape;
    (void)input_data;
    (void)filter_shape;
    (void)filter_data;
    (void)bias_shape;
    (void)bias_data;
    (void)output_shape;
    (void)output_data;
  }
}

template <>
inline void
FullyConnected<int8_t>(const tflite::FullyConnectedParams &params,
                       const tflite::RuntimeShape &input_shape, const int8_t *input_data,
                       const tflite::RuntimeShape &filter_shape, const int8_t *filter_data,
                       const tflite::RuntimeShape &bias_shape, const int32_t *bias_data,
                       const tflite::RuntimeShape &output_shape, int8_t *output_data)
{
  tflite::reference_integer_ops::FullyConnected(params, input_shape, input_data, filter_shape,
                                                filter_data, bias_shape, bias_data, output_shape,
                                                output_data);
}
} // namespace luci_interpreter_pal

#endif // LUCI_INTERPRETER_PAL_FULLYCONNECTED_H
