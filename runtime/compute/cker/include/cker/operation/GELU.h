/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __NNFW_CKER_GELU_H__
#define __NNFW_CKER_GELU_H__

#include "cker/Shape.h"
#include "cker/Types.h"
#include "cker/eigen/Utils.h"
#include <Eigen/Core>

#include <cmath>

namespace nnfw
{
namespace cker
{

namespace gelu_internal
{

constexpr float kSqrt2dPi = M_2_SQRTPI * M_SQRT1_2; // sqrt( 2 / pi )

} // namespace gelu_internal

inline void GELU(const GELUParams &params, const Shape &input_shape, const float *input_data,
                 const Shape &output_shape, float *output_data)
{
  const auto input_map = MapAsVector(input_data, input_shape);
  auto output_map = MapAsVector(output_data, output_shape);

  if (params.approximate)
  {
    // 0.5 * x * ( 1 + tanh( sqrt( 2 / pi ) * ( x + 0.044715 * x^3 ) ) )
    output_map.array() = 0.5f * input_map.array() *
                         (1.0f + (gelu_internal::kSqrt2dPi *
                                  (input_map.array() + 0.044715f * input_map.array().cube()))
                                   .tanh());
  }
  else
  {
    // Note: 0.5 * x * ( 1 + erf( x / sqrt( 2 ) ) ) is commonly used, but cause
    // catastropic cancellation for large negative inputs. Rewriting the
    // expression via erfc avoids the numerical stability issues.
    const float neg_sqrt1_2 = -static_cast<float>(M_SQRT1_2);
    auto x = input_map.array();
    auto x_scaled = x * neg_sqrt1_2;
    auto erfc_x_scaled = x_scaled.matrix().unaryExpr([](float val) { return erfcf(val); });
    output_map.array() = 0.5f * x * erfc_x_scaled.array();
  }
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_GELU_H__
