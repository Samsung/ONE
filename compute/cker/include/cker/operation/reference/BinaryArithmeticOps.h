/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __NNFW_CKER_REFERENCE_BINARYARITHMETICOPS_H__
#define __NNFW_CKER_REFERENCE_BINARYARITHMETICOPS_H__

#include "cker/Shape.h"
#include "cker/Utils.h"

#include <cmath>

namespace nnfw
{
namespace cker
{
namespace reference
{

template <typename T>
inline void BinaryArithmeticOp(const BinaryArithmeticOpParam &params, const Shape &input1_shape,
                               const T *input1_data, const Shape &input2_shape,
                               const T *input2_data, const Shape &output_shape, T *output_data,
                               const std::function<T(const T &, const T &)> &fn)
{
  const int32_t flat_size = MatchingFlatSize(input1_shape, input2_shape, output_shape);
  for (int i = 0; i < flat_size; ++i)
  {
    output_data[i] = ActivationFunctionWithMinMax(fn(input1_data[i], input2_data[i]),
                                                  params.quantized_activation_min,
                                                  params.quantized_activation_max);
  }
}

template <>
inline void BinaryArithmeticOp(const BinaryArithmeticOpParam &params, const Shape &input1_shape,
                               const float *input1_data, const Shape &input2_shape,
                               const float *input2_data, const Shape &output_shape,
                               float *output_data,
                               const std::function<float(const float &, const float &)> &fn)
{
  const int size = MatchingFlatSize(input1_shape, input2_shape, output_shape);
  for (int i = 0; i < size; i++)
  {
    output_data[i] =
        ActivationFunctionWithMinMax(fn(input1_data[i], input2_data[i]),
                                     params.float_activation_min, params.float_activation_max);
  }
}

} // namespace reference
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_REFERENCE_BINARYARITHMETICOPS_H__
