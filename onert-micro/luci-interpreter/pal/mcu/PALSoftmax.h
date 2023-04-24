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

#ifndef LUCI_INTERPRETER_PAL_SOFTMAX_H
#define LUCI_INTERPRETER_PAL_SOFTMAX_H

#include <tensorflow/lite/kernels/internal/reference/softmax.h>

namespace luci_interpreter_pal
{
static inline void PopulateSoftmaxLookupTable(tflite::SoftmaxParams *data, float input_scale,
                                              float beta)
{
  // Do nothing for mcu
  (void)data;
  (void)input_scale;
  (void)beta;
}

static inline void InitializeParams(tflite::SoftmaxParams *params, float input_scale, float beta)
{
  // TODO Impl it
  assert(false && "Softmax NYI");
  (void)params;
  (void)input_scale;
  (void)beta;
}

template <typename T>
static inline void Softmax(const tflite::SoftmaxParams &params,
                           const tflite::RuntimeShape &input_shape, const T *input_data,
                           const tflite::RuntimeShape &output_shape, T *output_data)
{
  // TODO Impl it
  // MARK: At this moment this operation doesn't support on mcu
  assert(false && "Softmax NYI");
  (void)params;
  (void)input_shape;
  (void)input_data;
  (void)output_shape;
  (void)output_data;
}
} // namespace luci_interpreter_pal

#endif // LUCI_INTERPRETER_PAL_SOFTMAX_H
