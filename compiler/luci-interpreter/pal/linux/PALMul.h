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

#ifndef LUCI_INTERPRETER_PAL_MUL_H
#define LUCI_INTERPRETER_PAL_MUL_H

#include <tensorflow/lite/kernels/internal/optimized/optimized_ops.h>

namespace luci_interpreter_pal
{
static inline void Mul(tflite::ArithmeticParams& params,
                       const tflite::RuntimeShape& input1_shape, const float* input1_data,
                       const tflite::RuntimeShape& input2_shape, const float* input2_data,
                       const tflite::RuntimeShape& output_shape, float* output_data)
{
  tflite::optimized_ops::Mul(params, input1_shape, input1_data, input2_shape,
                             input2_data, output_shape, output_data);
}

static inline void BroadcastMul4DSlow(tflite::ArithmeticParams& params,
                                      const tflite::RuntimeShape& input1_shape, const float* input1_data,
                                      const tflite::RuntimeShape& input2_shape, const float* input2_data,
                                      const tflite::RuntimeShape& output_shape, float* output_data)
{
  tflite::optimized_ops::BroadcastMul4DSlow(params, input1_shape, input1_data, input2_shape,
                                            input2_data, output_shape, output_data);
}
}

#endif // LUCI_INTERPRETER_PAL_MUL_H