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
#include <arm_nnfunctions.h>

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
  assert(output_shape.DimensionsCount() == 2);

  const int batches = output_shape.Dims(0);
  const int output_depth = output_shape.Dims(1);

  const int filter_dim_count = filter_shape.DimensionsCount();
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);

  cmsis_nn_fc_params fc_params;
  fc_params.input_offset = params.input_offset;
  fc_params.output_offset = params.output_offset;
  fc_params.filter_offset = params.weights_offset;
  fc_params.activation.min = params.quantized_activation_min;
  fc_params.activation.max = params.quantized_activation_max;

  cmsis_nn_per_tensor_quant_params quant_params;
  quant_params.multiplier = params.output_multiplier;
  quant_params.shift = params.output_shift;

  cmsis_nn_dims input_dims;
  input_dims.n = batches;
  input_dims.h = 1;
  input_dims.w = 1;
  input_dims.c = accum_depth;

  cmsis_nn_dims filter_dims;
  filter_dims.n = accum_depth;
  filter_dims.h = 1;
  filter_dims.w = 1;
  filter_dims.c = output_depth;

  cmsis_nn_dims bias_dims;
  bias_dims.n = 1;
  bias_dims.h = 1;
  bias_dims.w = 1;
  bias_dims.c = output_depth;

  cmsis_nn_dims output_dims;
  output_dims.n = batches;
  output_dims.h = 1;
  output_dims.w = 1;
  output_dims.c = output_depth;

  int32_t buf_size = arm_fully_connected_s8_get_buffer_size(&filter_dims);
  auto buffer = std::make_unique<int8_t[]>(buf_size);
  assert(buffer != nullptr);

  cmsis_nn_context ctx;
  ctx.buf = buffer.get();
  ctx.size = buf_size;

  auto res =
    arm_fully_connected_s8(&ctx, &fc_params, &quant_params, &input_dims, input_data, &filter_dims,
                           filter_data, &bias_dims, bias_data, &output_dims, output_data);
  assert(res == ARM_MATH_SUCCESS);
}
} // namespace luci_interpreter_pal

#endif // LUCI_INTERPRETER_PAL_FULLYCONNECTED_H
