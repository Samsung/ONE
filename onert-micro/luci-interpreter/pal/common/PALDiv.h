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

#ifndef LUCI_INTERPRETER_PAL_DIV_COMMON_H
#define LUCI_INTERPRETER_PAL_DIV_COMMON_H

#include "PALArithmeticOpCommon.h"

namespace luci_interpreter_pal
{
template <typename T>
inline void Div(const ArithmeticParams &params, const int flat_size, const T *input1_data,
                const T *input2_data, T *output_data)
{
  ArithmeticOp<T, DivFn<T>>(params, flat_size, input1_data, input2_data, output_data);
}

template <typename T>
inline void DivScalar(const ArithmeticParams &params, const int flat_size, const T *input_data,
                      const T scalar_value, T *output_data)
{
  T activation_min, activation_max;
  getActivationParams(params, &activation_min, &activation_max);

  for (int i = 0; i < flat_size; ++i)
    output_data[i] =
      std::min(std::max(input_data[i] / scalar_value, activation_min), activation_max);
}

template <typename T>
inline void
BroadcastDiv4DSlow(const ArithmeticParams &params,
                   const luci_interpreter::RuntimeShape &input1_shape, const T *input1_data,
                   const luci_interpreter::RuntimeShape &input2_shape, const T *input2_data,
                   const luci_interpreter::RuntimeShape &output_shape, T *output_data)
{
  const int flat_size = input1_shape.flatSize();

  if (params.broadcast_category == BroadcastableOpCategory::kScalarFirstBroadcast)
  {
    return DivScalar(params, flat_size, input2_data, input1_data[0], output_data);
  }
  else if (params.broadcast_category == BroadcastableOpCategory::kScalarSecondBroadcast)
  {
    return DivScalar(params, flat_size, input1_data, input2_data[0], output_data);
  }

  BroadcastArithmeticOp4DSlow<T, DivFn<T>>(params, input1_shape, input1_data, input2_shape,
                                           input2_data, output_shape, output_data);
}

} // namespace luci_interpreter_pal

#endif // LUCI_INTERPRETER_PAL_DIV_COMMON_H
