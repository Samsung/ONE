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

#include <tensorflow/lite/kernels/internal/optimized/legacy_optimized_ops.h>
#include <tensorflow/lite/kernels/internal/reference/integer_ops/conv.h>

namespace luci_interpreter_pal
{
static inline void Conv(const tflite::ConvParams &params, const tflite::RuntimeShape &input_shape,
                        const float *input_data, const tflite::RuntimeShape &filter_shape,
                        const float *filter_data, const tflite::RuntimeShape &bias_shape,
                        const float *bias_data, const tflite::RuntimeShape &output_shape,
                        float *output_data, const tflite::RuntimeShape &im2col_shape,
                        float *im2col_data)
{
  if (im2col_data)
  {
    tflite::optimized_ops::Conv(params, input_shape, input_data, filter_shape, filter_data,
                                bias_shape, bias_data, output_shape, output_data, im2col_shape,
                                im2col_data);
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
                        uint8 *output_data, const tflite::RuntimeShape &im2col_shape,
                        uint8 *im2col_data)
{
  // TODO This should only be done once (although it takes only a few microseconds).
  //  Also, the user should be able to adjust the number of threads.
  auto gemmlowp_context = std::make_unique<gemmlowp::GemmContext>();
  gemmlowp_context->set_max_num_threads(static_cast<int>(std::thread::hardware_concurrency()));

  tflite::reference_ops::Conv(params, input_shape, input_data, filter_shape, filter_data,
                              bias_shape, bias_data, output_shape, output_data, im2col_shape,
                              im2col_data, gemmlowp_context.get());
}

static inline void ConvPerChannel(const tflite::ConvParams &params, const int32_t *mult,
                                  const int32_t *shifts, const tflite::RuntimeShape &input_shape,
                                  const int8 *input_data, const tflite::RuntimeShape &filter_shape,
                                  const int8 *filter_data, const tflite::RuntimeShape &bias_shape,
                                  const int32 *bias_data, const tflite::RuntimeShape &output_shape,
                                  int8 *output_data, const tflite::RuntimeShape &im2col_shape,
                                  int8 *im2col_data)
{
  // TODO enable optimized version
  tflite::reference_integer_ops::ConvPerChannel(params, mult, shifts, input_shape, input_data,
                                                filter_shape, filter_data, bias_shape, bias_data,
                                                output_shape, output_data);
}

} // namespace luci_interpreter_pal

#endif // LUCI_INTERPRETER_PAL_CONV2D_H
