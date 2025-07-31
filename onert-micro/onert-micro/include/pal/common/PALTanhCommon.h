/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

#ifndef ONERT_MICRO_PAL_TANH_COMMON_H
#define ONERT_MICRO_PAL_TANH_COMMON_H

#include "PALUnaryOpCommon.h"
#include <cmath>

namespace onert_micro
{
namespace execute
{
namespace pal
{
template <typename T>
inline OMStatus Tanh(const core::OMRuntimeShape &input_shape, const T *input_data,
                     const core::OMRuntimeShape &output_shape, T *output_data)
{
  const uint32_t flat_size = input_shape.flatSize();

  if (flat_size == -1)
    return UnknownError;

  assert(input_data != nullptr);
  assert(output_data != nullptr);

  assert(input_shape == output_shape);

  for (int i = 0; i < flat_size; i++)
  {
    output_data[i] = std::tanh(input_data[i]);
  }

  return Ok;
}

template <typename T>
inline OMStatus QuantizedTanh(const core::OMRuntimeShape &input_shape,
                              const onert_micro::core::QuantizationParams &input_qparams,
                              const T *input_data, const core::OMRuntimeShape &output_shape,
                              const onert_micro::core::QuantizationParams &output_qparams,
                              T *output_data)
{
  const uint32_t flat_size = input_shape.flatSize();
  for (int i = 0; i < flat_size; i++)
  {
    // Dequantize.
    float val = static_cast<float>((input_data[i] - static_cast<T>(input_qparams.zero_point)) *
                                   input_qparams.scale);
    // float result
    float result = std::tanh(val);

    // Quantize float to output type
    result = result / output_qparams.scale + output_qparams.zero_point;
    result = std::max<float>(std::numeric_limits<T>::min(), result);
    result = std::min<float>(std::numeric_limits<T>::max(), result);

    output_data[i] = static_cast<T>(result);
  }

  return Ok;
}

} // namespace pal
} // namespace execute
} // namespace onert_micro

#endif // ONERT_MICRO_PAL_TANH_COMMON_H
