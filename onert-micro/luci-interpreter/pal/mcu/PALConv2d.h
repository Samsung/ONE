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

#ifndef LUCI_INTERPRETER_PAL_CONV2D_H
#define LUCI_INTERPRETER_PAL_CONV2D_H

#include <tensorflow/lite/kernels/internal/reference/conv.h>
#include <tensorflow/lite/kernels/internal/reference/integer_ops/conv.h>

namespace luci_interpreter_pal
{
static inline void Conv(const tflite::ConvParams &params, const tflite::RuntimeShape &input_shape,
                        const float *input_data, const tflite::RuntimeShape &filter_shape,
                        const float *filter_data, const tflite::RuntimeShape &bias_shape,
                        const float *bias_data, const tflite::RuntimeShape &output_shape,
                        float *output_data, const tflite::RuntimeShape &scratchpad_shape,
                        float *scratchpad_data)
{
  (void)scratchpad_shape;
  (void)scratchpad_data;
  tflite::reference_ops::Conv(params, input_shape, input_data, filter_shape, filter_data,
                              bias_shape, bias_data, output_shape, output_data,
                              tflite::RuntimeShape(), nullptr);
}

static inline void Conv(const tflite::ConvParams &params, const tflite::RuntimeShape &input_shape,
                        const uint8_t *input_data, const tflite::RuntimeShape &filter_shape,
                        const uint8_t *filter_data, const tflite::RuntimeShape &bias_shape,
                        const int32_t *bias_data, const tflite::RuntimeShape &output_shape,
                        uint8_t *output_data, const tflite::RuntimeShape &scratchpad_shape,
                        uint8_t *scratchpad_data)
{
  (void)scratchpad_shape;
  (void)scratchpad_data;
  tflite::reference_ops::Conv(params, input_shape, input_data, filter_shape, filter_data,
                              bias_shape, bias_data, output_shape, output_data, scratchpad_shape,
                              scratchpad_data, nullptr);
}

static inline void
ConvPerChannel(const tflite::ConvParams &params, const int32_t *mult, const int32_t *shifts,
               const tflite::RuntimeShape &input_shape, const int8_t *input_data,
               const tflite::RuntimeShape &filter_shape, const int8_t *filter_data,
               const tflite::RuntimeShape &bias_shape, const int32_t *bias_data,
               const tflite::RuntimeShape &output_shape, int8_t *output_data,
               const tflite::RuntimeShape &scratchpad_shape, int8_t *scratchpad_data)
{
  (void)scratchpad_shape;
  (void)scratchpad_data;
  tflite::reference_integer_ops::ConvPerChannel(params, mult, shifts, input_shape, input_data,
                                                filter_shape, filter_data, bias_shape, bias_data,
                                                output_shape, output_data);
}

static inline void SetupScratchpadTensor(luci_interpreter::Tensor *scratchpad,
                                         const luci_interpreter::DataType &input_data_type,
                                         const tflite::ConvParams &params,
                                         const tflite::RuntimeShape &input_shape,
                                         const tflite::RuntimeShape &filter_shape,
                                         const tflite::RuntimeShape &output_shape)
{
  (void)input_data_type;
  (void)params;
  (void)input_shape;
  (void)filter_shape;
  (void)output_shape;
  (void)scratchpad;
}

} // namespace luci_interpreter_pal

#endif // LUCI_INTERPRETER_PAL_CONV2D_H
