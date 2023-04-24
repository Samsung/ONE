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

#ifndef LUCI_INTERPRETER_PAL_AVERAGEPOOL2D_H
#define LUCI_INTERPRETER_PAL_AVERAGEPOOL2D_H

#include <tensorflow/lite/kernels/internal/reference/integer_ops/pooling.h>
#include <tensorflow/lite/kernels/internal/reference/pooling.h>
#include <arm_nn_types.h>
#include <arm_nnfunctions.h>

namespace luci_interpreter_pal
{
template <typename T>
static inline void AveragePool(const tflite::PoolParams &params,
                               const tflite::RuntimeShape &input_shape, const T *input_data,
                               const tflite::RuntimeShape &output_shape, T *output_data,
                               const tflite::RuntimeShape &scratchpad_shape, T *scratchpad_data)
{
  {
    // MARK: At this moment this operation is not supported
    assert(false && "AveragePool NYI");
    (void)params;
    (void)input_shape;
    (void)input_data;
    (void)output_shape;
    (void)output_data;
    (void)scratchpad_shape;
    (void)scratchpad_data;
  }
}

template <>
inline void AveragePool<int8_t>(const tflite::PoolParams &params,
                                const tflite::RuntimeShape &input_shape, const int8_t *input_data,
                                const tflite::RuntimeShape &output_shape, int8_t *output_data,
                                const tflite::RuntimeShape &scratchpad_shape,
                                int8_t *scratchpad_data)
{
  assert(input_shape.DimensionsCount() == 4);
  assert(output_shape.DimensionsCount() == 4);
  assert(scratchpad_data != nullptr);

  const int32_t batches = tflite::MatchingDim(input_shape, 0, output_shape, 0);
  assert(batches == 1);

  const int depth = tflite::MatchingDim(input_shape, 3, output_shape, 3);

  cmsis_nn_dims input_dims;
  input_dims.n = 1;
  input_dims.h = input_shape.Dims(1);
  input_dims.w = input_shape.Dims(2);
  input_dims.c = depth;

  cmsis_nn_dims output_dims;
  output_dims.n = 1;
  output_dims.h = output_shape.Dims(1);
  output_dims.w = output_shape.Dims(2);
  output_dims.c = depth;

  cmsis_nn_pool_params pool_params;
  pool_params.stride.h = params.stride_height;
  pool_params.stride.w = params.stride_width;
  pool_params.padding.h = params.padding_values.height;
  pool_params.padding.w = params.padding_values.width;
  pool_params.activation.min = params.quantized_activation_min;
  pool_params.activation.max = params.quantized_activation_max;

  cmsis_nn_dims filter_dims;
  filter_dims.n = 1;
  filter_dims.h = params.filter_height;
  filter_dims.w = params.filter_width;
  filter_dims.c = 1;

  cmsis_nn_context ctx;
  ctx.buf = scratchpad_data;
  ctx.size = scratchpad_shape.Dims(0);
  auto res = arm_avgpool_s8(&ctx, &pool_params, &input_dims, input_data, &filter_dims, &output_dims,
                            output_data);
  assert(res == ARM_MATH_SUCCESS);
}

static inline void SetupScratchpadTensor(luci_interpreter::Tensor *scratchpad,
                                         const luci_interpreter::DataType &input_data_type,
                                         const tflite::RuntimeShape &input_shape,
                                         const tflite::RuntimeShape &output_shape)

{
  if (input_data_type == luci_interpreter::DataType::S8)
  {
    assert(input_shape.DimensionsCount() == 4);
    assert(output_shape.DimensionsCount() == 4);

    const int32_t output_width = output_shape.Dims(2);
    const int32_t depth = tflite::MatchingDim(input_shape, 3, output_shape, 3);

    const int32_t buf_size = arm_avgpool_s8_get_buffer_size(output_width, depth);
    auto data_type_size = static_cast<int32_t>(luci_interpreter::getDataTypeSize(input_data_type));

    luci_interpreter::Shape scratchpad_shape{buf_size * data_type_size};
    scratchpad->resize(scratchpad_shape);
  }
  else
  {
    scratchpad->set_allocatable(false);
  }
}

} // namespace luci_interpreter_pal

#endif // LUCI_INTERPRETER_PAL_AVERAGEPOOL2D_H
