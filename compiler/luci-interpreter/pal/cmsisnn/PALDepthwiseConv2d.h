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

#ifndef LUCI_INTERPRETER_PAL_DEPTHWISECONV2D_H
#define LUCI_INTERPRETER_PAL_DEPTHWISECONV2D_H

#include <tensorflow/lite/kernels/internal/reference/depthwiseconv_float.h>
#include <tensorflow/lite/kernels/internal/reference/depthwiseconv_uint8.h>
#include <tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h>
#include <arm_nnfunctions.h>

namespace luci_interpreter_pal
{
template <typename T>
static inline void
DepthwiseConvPerChannel(const tflite::DepthwiseParams &params, const int32_t *output_multiplier,
                        const int32_t *output_shift, const tflite::RuntimeShape &input_shape,
                        const T *input_data, const tflite::RuntimeShape &filter_shape,
                        const T *filter_data, const tflite::RuntimeShape &bias_shape,
                        const int32_t *bias_data, const tflite::RuntimeShape &output_shape,
                        T *output_data)
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
  }
}

template <>
inline void
DepthwiseConvPerChannel<int8_t>(const tflite::DepthwiseParams &params,
                                const int32_t *output_multiplier, const int32_t *output_shift,
                                const tflite::RuntimeShape &input_shape, const int8_t *input_data,
                                const tflite::RuntimeShape &filter_shape, const int8_t *filter_data,
                                const tflite::RuntimeShape &bias_shape, const int32_t *bias_data,
                                const tflite::RuntimeShape &output_shape, int8_t *output_data)
{
  cmsis_nn_dw_conv_params dw_conv_params;
  dw_conv_params.dilation.h = params.dilation_height_factor;
  dw_conv_params.dilation.w = params.dilation_width_factor;

  if (dw_conv_params.dilation.h == 1 && dw_conv_params.dilation.w == 1)
  {
    dw_conv_params.input_offset = params.input_offset;
    dw_conv_params.output_offset = params.output_offset;
    dw_conv_params.stride.h = params.stride_height;
    dw_conv_params.stride.w = params.stride_width;
    dw_conv_params.padding.h = params.padding_values.height;
    dw_conv_params.padding.w = params.padding_values.width;

    dw_conv_params.activation.min = params.quantized_activation_min;
    dw_conv_params.activation.max = params.quantized_activation_max;
    dw_conv_params.ch_mult = params.depth_multiplier;

    cmsis_nn_per_channel_quant_params quant_params;
    int32_t output_multiplier = params.output_multiplier;
    int32_t output_shift = params.output_shift;

    quant_params.multiplier = &output_multiplier;
    quant_params.shift = &output_shift;

    assert(dw_conv_params.activation.min <= dw_conv_params.activation.max);
    const int batch_size = tflite::MatchingDim(input_shape, 0, output_shape, 0);
    const int output_depth = tflite::MatchingDim(filter_shape, 3, output_shape, 3);
    if (bias_data)
    {
      assert(bias_shape.FlatSize() == output_depth);
    }

    cmsis_nn_dims input_dims;
    input_dims.n = batch_size;
    input_dims.h = input_shape.Dims(1);
    input_dims.w = input_shape.Dims(2);
    input_dims.c = input_shape.Dims(3);

    cmsis_nn_dims filter_dims;
    filter_dims.n = filter_shape.Dims(0);
    filter_dims.h = filter_shape.Dims(1);
    filter_dims.w = filter_shape.Dims(2);
    filter_dims.c = output_depth;

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

    const int32_t buf_size = arm_depthwise_conv_wrapper_s8_get_buffer_size(
      &dw_conv_params, &input_dims, &filter_dims, &output_dims);

    auto buffer = std::make_unique<int8_t[]>(buf_size);
    assert(buffer != nullptr);

    cmsis_nn_context ctx;
    ctx.buf = buffer.get();
    ctx.size = buf_size;

    auto res = arm_depthwise_conv_wrapper_s8(&ctx, &dw_conv_params, &quant_params, &input_dims,
                                             input_data, &filter_dims, filter_data, &bias_dims,
                                             bias_data, &output_dims, output_data);
    assert(res == ARM_MATH_SUCCESS);
  }
  else
  {
    tflite::reference_integer_ops::DepthwiseConvPerChannel(
      params, output_multiplier, output_shift, input_shape, input_data, filter_shape, filter_data,
      bias_shape, bias_data, output_shape, output_data);
  }
}
} // namespace luci_interpreter_pal

#endif // LUCI_INTERPRETER_PAL_DEPTHWISECONV2D_H
