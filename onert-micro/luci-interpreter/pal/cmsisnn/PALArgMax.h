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

#ifndef LUCI_INTERPRETER_PAL_ARGMAX_H
#define LUCI_INTERPRETER_PAL_ARGMAX_H

#include <tensorflow/lite/kernels/internal/reference/arg_min_max.h>

namespace luci_interpreter_pal
{
template <typename T1, typename T2, typename T3>
static inline void ArgMinMax(const tflite::RuntimeShape &input1_shape, const T1 *input1_data,
                             const T2 *axis, const tflite::RuntimeShape &output_shape,
                             T3 *output_data, const std::greater<T1> cmp)
{
  tflite::reference_ops::ArgMinMax(input1_shape, input1_data, axis, output_shape, output_data, cmp);
}
} // namespace luci_interpreter_pal

#endif // LUCI_INTERPRETER_PAL_ARGMAX_H
