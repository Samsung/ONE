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

#include <tensorflow/lite/kernels/internal/optimized/legacy_optimized_ops.h>
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
  if (scratchpad_data)
  {
    const int32_t batches = tflite::MatchingDim(input_shape, 0, output_shape, 0);
    const int32_t input_depth = tflite::MatchingDim(input_shape, 3, filter_shape, 3);
    const int32_t output_height = output_shape.Dims(1);
    const int32_t output_width = output_shape.Dims(2);
    const int32_t filter_height = filter_shape.Dims(1);
    const int32_t filter_width = filter_shape.Dims(2);
    tflite::RuntimeShape im2col_shape{batches, output_height, output_width,
                                      input_depth * filter_height * filter_width};

    tflite::optimized_ops::Conv(params, input_shape, input_data, filter_shape, filter_data,
                                bias_shape, bias_data, output_shape, output_data, im2col_shape,
                                scratchpad_data);
  }
  else
    tflite::reference_ops::Conv(params, input_shape, input_data, filter_shape, filter_data,
                                bias_shape, bias_data, output_shape, output_data,
                                tflite::RuntimeShape(), nullptr);
}

static inline void Conv(const tflite::ConvParams &params, const tflite::RuntimeShape &input_shape,
                        const uint8 *input_data, const tflite::RuntimeShape &filter_shape,
                        const uint8 *filter_data, const tflite::RuntimeShape &bias_shape,
                        const int32 *bias_data, const tflite::RuntimeShape &output_shape,
                        uint8 *output_data, const tflite::RuntimeShape &scratchpad_shape,
                        uint8 *scratchpad_data)
{
  // TODO This should only be done once (although it takes only a few microseconds).
  //  Also, the user should be able to adjust the number of threads.
  auto gemmlowp_context = std::make_unique<gemmlowp::GemmContext>();
  gemmlowp_context->set_max_num_threads(static_cast<int>(std::thread::hardware_concurrency()));

  tflite::reference_ops::Conv(params, input_shape, input_data, filter_shape, filter_data,
                              bias_shape, bias_data, output_shape, output_data, scratchpad_shape,
                              scratchpad_data, gemmlowp_context.get());
}

static inline void ConvPerChannel(const tflite::ConvParams &params, const int32_t *mult,
                                  const int32_t *shifts, const tflite::RuntimeShape &input_shape,
                                  const int8 *input_data, const tflite::RuntimeShape &filter_shape,
                                  const int8 *filter_data, const tflite::RuntimeShape &bias_shape,
                                  const int32 *bias_data, const tflite::RuntimeShape &output_shape,
                                  int8 *output_data, const tflite::RuntimeShape &scratchpad_shape,
                                  int8 *scratchpad_data)
{
  (void)scratchpad_shape;
  (void)scratchpad_data;
  // TODO enable optimized version
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
  const int32_t filter_height = filter_shape.Dims(1);
  const int32_t filter_width = filter_shape.Dims(2);

  // Allocate tensor for scratchpad, if needed.
  // The checks here should be aligned with the actual implementation.
  const bool need_dilated_scratchpad =
    params.dilation_height_factor != 1 || params.dilation_width_factor != 1;
  const bool need_non_dilated_scratchpad = params.stride_height != 1 || params.stride_width != 1 ||
                                           filter_height != 1 || filter_width != 1;
  auto _need_scratchpad = input_data_type != luci_interpreter::DataType::S16 &&
                          (need_dilated_scratchpad || need_non_dilated_scratchpad);

  if (_need_scratchpad)
  {
    const int32_t batches = tflite::MatchingDim(input_shape, 0, output_shape, 0);
    const int32_t input_depth = tflite::MatchingDim(input_shape, 3, filter_shape, 3);
    const int32_t output_height = output_shape.Dims(1);
    const int32_t output_width = output_shape.Dims(2);

    auto data_type_size = static_cast<int32_t>(luci_interpreter::getDataTypeSize(input_data_type));
    int32_t scratchpad_size = batches * output_width * output_height * input_depth * filter_height *
                              filter_width * data_type_size;
    luci_interpreter::Shape scratchpad_shape{scratchpad_size};
    scratchpad->resize(scratchpad_shape);
  }
  else
  {
    scratchpad->set_allocatable(false);
  }
}

} // namespace luci_interpreter_pal

#endif // LUCI_INTERPRETER_PAL_CONV2D_H
