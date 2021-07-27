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

#ifndef LUCI_INTERPRETER_PAL_LOGSOFTMAX_H
#define LUCI_INTERPRETER_PAL_LOGSOFTMAX_H

#include <tensorflow/lite/kernels/internal/reference/reference_ops.h>


namespace luci_interpreter_pal
{
static inline void PopulateSoftmaxLookupTable(tflite::SoftmaxParams* data, float input_scale,
                                              float beta)
{
  //Do nothing for MCU
}

static inline void InitializeParams(tflite::SoftmaxParams* params, float input_scale, float beta)
{
  static const int kScaledDiffIntegerBits = 5;
  tflite::PreprocessLogSoftmaxScalingExp(
    beta, input_scale, kScaledDiffIntegerBits, &params->input_multiplier,
    &params->input_left_shift, &params->reverse_scaling_divisor,
    &params->reverse_scaling_right_shift);
  params->reverse_scaling_right_shift *= -1;
  params->diff_min = -tflite::CalculateInputRadius(kScaledDiffIntegerBits,
                                                   params->input_left_shift);
}

static inline void LogSoftmax(tflite::SoftmaxParams& params, float input_scale,
                       const tflite::RuntimeShape& input_shape, const uint8* input_data,
                       const tflite::RuntimeShape& output_shape, uint8* output_data)
{
  tflite::reference_ops::LogSoftmax(params, input_shape, input_data,
                                    output_shape, output_data);
}
}

#endif // LUCI_INTERPRETER_PAL_LOGSOFTMAX_H
