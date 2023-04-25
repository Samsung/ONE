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

#ifndef LUCI_INTERPRETER_PAL_DEPTHWISECONV2D_H
#define LUCI_INTERPRETER_PAL_DEPTHWISECONV2D_H

#include <tensorflow/lite/kernels/internal/reference/depthwiseconv_float.h>
#include <tensorflow/lite/kernels/internal/reference/depthwiseconv_uint8.h>
#include <tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h>

namespace luci_interpreter_pal
{
template <typename T>
static inline void
DepthwiseConvPerChannel(const tflite::DepthwiseParams &params, const int32_t *output_multiplier,
                        const int32_t *output_shift, const tflite::RuntimeShape &input_shape,
                        const T *input_data, const tflite::RuntimeShape &filter_shape,
                        const T *filter_data, const tflite::RuntimeShape &bias_shape,
                        const int32_t *bias_data, const tflite::RuntimeShape &output_shape,
                        T *output_data, const tflite::RuntimeShape &scratchpad_shape,
                        T *scratchpad_data)
{
  {
    // MARK: At this moment this operation is not supported
    assert(false && "DepthwiseConvPerChannel NYI");
    (void)params;
    (void)output_multiplier;
    (void)output_shift;
    (void)input_shape;
    (void)output_data;
    (void)input_data;
    (void)filter_shape;
    (void)filter_data;
    (void)bias_shape;
    (void)bias_data;
    (void)output_shape;
    (void)output_data;
    (void)scratchpad_shape;
    (void)scratchpad_data;
  }
}

template <>
inline void DepthwiseConvPerChannel<int8_t>(
  const tflite::DepthwiseParams &params, const int32_t *output_multiplier,
  const int32_t *output_shift, const tflite::RuntimeShape &input_shape, const int8_t *input_data,
  const tflite::RuntimeShape &filter_shape, const int8_t *filter_data,
  const tflite::RuntimeShape &bias_shape, const int32_t *bias_data,
  const tflite::RuntimeShape &output_shape, int8_t *output_data,
  const tflite::RuntimeShape &scratchpad_shape, int8_t *scratchpad_data)
{
  (void)scratchpad_shape;
  (void)scratchpad_data;
  tflite::reference_integer_ops::DepthwiseConvPerChannel(
    params, output_multiplier, output_shift, input_shape, input_data, filter_shape, filter_data,
    bias_shape, bias_data, output_shape, output_data);
}

static inline void SetupScratchpadTensor(luci_interpreter::Tensor *scratchpad,
                                         const tflite::DepthwiseParams &params,
                                         const luci_interpreter::DataType &input_data_type,
                                         const tflite::RuntimeShape &input_shape,
                                         const tflite::RuntimeShape &filter_shape,
                                         const tflite::RuntimeShape &output_shape)

{
  (void)params;
  (void)input_data_type;
  (void)input_shape;
  (void)filter_shape;
  (void)output_shape;

  scratchpad->set_allocatable(false);
}

} // namespace luci_interpreter_pal

#endif // LUCI_INTERPRETER_PAL_DEPTHWISECONV2D_H
