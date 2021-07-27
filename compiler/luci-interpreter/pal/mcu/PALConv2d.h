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

#include <tensorflow/lite/kernels/internal/reference/reference_ops.h>

namespace luci_interpreter_pal
{
static inline void Conv(const tflite::ConvParams& params, const tflite::RuntimeShape& input_shape,
                        const float* input_data, const tflite::RuntimeShape& filter_shape,
                        const float* filter_data, const tflite::RuntimeShape& bias_shape,
                        const float* bias_data, const tflite::RuntimeShape& output_shape,
                        float* output_data, const tflite::RuntimeShape& im2col_shape,
                        float* im2col_data)
{
  tflite::reference_ops::Conv(params, input_shape, input_data, filter_shape,
                              filter_data, bias_shape, bias_data,
                              output_shape, output_data, tflite::RuntimeShape(), nullptr);
}

static inline void Conv(const tflite::ConvParams& params, const tflite::RuntimeShape& input_shape,
                        const uint8* input_data, const tflite::RuntimeShape& filter_shape,
                        const uint8* filter_data, const tflite::RuntimeShape& bias_shape,
                        const int32* bias_data, const tflite::RuntimeShape& output_shape,
                        uint8* output_data, const tflite::RuntimeShape& im2col_shape,
                        uint8* im2col_data)
{
 tflite::reference_ops::Conv(params, input_shape, input_data, filter_shape,
                              filter_data, bias_shape, bias_data,
                              output_shape, output_data, im2col_shape,
                              im2col_data, nullptr);
}

}

#endif // LUCI_INTERPRETER_PAL_CONV2D_H
