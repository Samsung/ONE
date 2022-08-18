/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef LUCI_INTERPRETER_PAL_QUANTIZE_H
#define LUCI_INTERPRETER_PAL_QUANTIZE_H

#include "PALreference_ops.h"

namespace luci_interpreter_pal
{
template <typename T>
static inline void Quantize(tflite::QuantizationParams &params,
                            const tflite::RuntimeShape &input_shape, const float *input_data,
                            const tflite::RuntimeShape &output_shape, T *output_data)
{
  tflite::reference_ops::AffineQuantize(params, input_shape, input_data, output_shape, output_data);
}

template <typename Input, typename Output>
static inline void Requantize(const Input *input_data, int32_t size,
                              int32_t effective_scale_multiplier, int32_t effective_scale_shift,
                              int32_t input_zero_point, int32_t output_zero_point,
                              Output *output_data)
{
  tflite::reference_ops::Requantize(input_data, size, effective_scale_multiplier,
                                    effective_scale_shift, input_zero_point, output_zero_point,
                                    output_data);
}
} // namespace luci_interpreter_pal

#endif // LUCI_INTERPRETER_PAL_QUANTIZE_H
