/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef LUCI_INTERPRETER_PAL_DEQUANTIZE_H
#define LUCI_INTERPRETER_PAL_DEQUANTIZE_H

#include <tensorflow/lite/kernels/internal/optimized/optimized_ops.h>

namespace luci_interpreter_pal
{
template <typename T>
static inline void Dequantize(tflite::DequantizationParams &params,
                              const tflite::RuntimeShape &input_shape, const T *input_data,
                              const tflite::RuntimeShape &output_shape, float *output_data)
{
  tflite::optimized_ops::Dequantize(params, input_shape, input_data, output_shape, output_data);
}

} // namespace luci_interpreter_pal

#endif // LUCI_INTERPRETER_PAL_DEQUANTIZE_H
