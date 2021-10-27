/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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
#include <arm_nn_types.h>
#include <arm_nnfunctions.h>

namespace luci_interpreter_pal
{
static inline void Conv(const tflite::ConvParams &params, const tflite::RuntimeShape &input_shape,
                        const float *input_data, const tflite::RuntimeShape &filter_shape,
                        const float *filter_data, const tflite::RuntimeShape &bias_shape,
                        const float *bias_data, const tflite::RuntimeShape &output_shape,
                        float *output_data, const tflite::RuntimeShape &im2col_shape,
                        float *im2col_data)
{
  (void)im2col_shape;
  (void)im2col_data;
  tflite::reference_ops::Conv(params, input_shape, input_data, filter_shape, filter_data,
                              bias_shape, bias_data, output_shape, output_data,
                              tflite::RuntimeShape(), nullptr);
}

static inline void Conv(const tflite::ConvParams &params, const tflite::RuntimeShape &input_shape,
                        const uint8 *input_data, const tflite::RuntimeShape &filter_shape,
                        const uint8 *filter_data, const tflite::RuntimeShape &bias_shape,
                        const int32 *bias_data, const tflite::RuntimeShape &output_shape,
                        uint8 *output_data, const tflite::RuntimeShape &im2col_shape,
                        uint8 *im2col_data)
{
  (void)im2col_shape;
  (void)im2col_data;
  tflite::reference_ops::Conv(params, input_shape, input_data, filter_shape, filter_data,
                              bias_shape, bias_data, output_shape, output_data, im2col_shape,
                              im2col_data, nullptr);
}

static inline void ConvPerChannel(const tflite::ConvParams &params, const int32_t *mult,
                                  const int32_t *shifts, const tflite::RuntimeShape &input_shape,
                                  const int8 *input_data, const tflite::RuntimeShape &filter_shape,
                                  const int8 *filter_data, const tflite::RuntimeShape &bias_shape,
                                  const int32 *bias_data, const tflite::RuntimeShape &output_shape,
                                  int8 *output_data, const tflite::RuntimeShape &im2col_shape,
                                  int8 *im2col_data)
{
  (void)im2col_shape;
  (void)im2col_data;

  cmsis_nn_conv_params conv_params;
  conv_params.dilation.h = params.dilation_height_factor;
  conv_params.dilation.w = params.dilation_width_factor;

  if (conv_params.dilation.h == 1 && conv_params.dilation.w == 1)
  {
    conv_params.input_offset = params.input_offset;
    conv_params.output_offset = params.output_offset;
    conv_params.stride.h = params.stride_height;
    conv_params.stride.w = params.stride_width;
    conv_params.padding.h = params.padding_values.height;
    conv_params.padding.w = params.padding_values.width;
    conv_params.activation.min = params.quantized_activation_min;
    conv_params.activation.max = params.quantized_activation_max;

    cmsis_nn_per_channel_quant_params quant_params;
    quant_params.multiplier = const_cast<int32_t *>(mult);
    quant_params.shift = const_cast<int32_t *>(shifts);

    assert(conv_params.activation.min <= conv_params.activation.max);
    assert(input_shape.DimensionsCount() == 4);
    assert(filter_shape.DimensionsCount() == 4);
    assert(output_shape.DimensionsCount() == 4);
    const int batch_size = tflite::MatchingDim(input_shape, 0, output_shape, 0);
    const int input_depth = tflite::MatchingDim(input_shape, 3, filter_shape, 3);
    const int output_depth = tflite::MatchingDim(filter_shape, 0, output_shape, 3);
    if (bias_data)
    {
      assert(bias_shape.FlatSize() == output_depth);
    }

    cmsis_nn_dims input_dims;
    input_dims.n = batch_size;
    input_dims.h = input_shape.Dims(1);
    input_dims.w = input_shape.Dims(2);
    input_dims.c = input_depth;

    cmsis_nn_dims filter_dims;
    filter_dims.n = output_depth;
    filter_dims.h = filter_shape.Dims(1);
    filter_dims.w = filter_shape.Dims(2);
    filter_dims.c = input_depth;

    cmsis_nn_dims bias_dims;
    bias_dims.n = 1;
    bias_dims.h = 1;
    bias_dims.w = 1;
    bias_dims.c = output_depth;

    cmsis_nn_dims output_dims;
    output_dims.n = batch_size;
    output_dims.h = output_shape.Dims(1);
    output_dims.w = output_shape.Dims(2);
    output_dims.c = output_depth;

    const int32_t buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims,
                                                                     &filter_dims, &output_dims);
    auto buffer = std::make_unique<int8_t[]>(buf_size);
    assert(buffer != nullptr);

    cmsis_nn_context ctx;
    ctx.buf = buffer.get();
    ctx.size = buf_size;

    auto res = arm_convolve_wrapper_s8(&ctx, &conv_params, &quant_params, &input_dims, input_data,
                                       &filter_dims, filter_data, &bias_dims, bias_data,
                                       &output_dims, output_data);
    assert(res == ARM_MATH_SUCCESS);
  }
  else
  {
    tflite::reference_integer_ops::ConvPerChannel(params, mult, shifts, input_shape, input_data,
                                                  filter_shape, filter_data, bias_shape, bias_data,
                                                  output_shape, output_data);
  }
}

} // namespace luci_interpreter_pal

#endif // LUCI_INTERPRETER_PAL_CONV2D_H
