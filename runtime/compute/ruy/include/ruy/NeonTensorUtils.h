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

#ifndef __NNFW_RUY_NEON_TENSOR_UTILS_H__
#define __NNFW_RUY_NEON_TENSOR_UTILS_H__

#include "ruy/neon/neon_check.h"

#ifdef USE_NEON

#define kFloatWeightsPerNeonLane 4

namespace nnfw
{
namespace ruy
{

inline bool NeonIsZeroVector(const float *vector, int v_size)
{
  // If v_size is not divisible by kFloatWeightsPerNeonLane, we cannot
  // use the main vectorized loop, and we need to process sequentially.
  // postamble_start shows the start index where this should happen.
  const int postamble_start = v_size - (v_size & (kFloatWeightsPerNeonLane - 1));

  const float32x4_t zero_x4_float = vmovq_n_f32(0.0f);
  for (int v = 0; v < postamble_start; v += kFloatWeightsPerNeonLane)
  {
    const float32x4_t i_x4_float = vld1q_f32(vector + v);
    uint32x4_t cmp_result = vceqq_f32(i_x4_float, zero_x4_float);
    if (vgetq_lane_u32(cmp_result, 0) == 0)
      return false;
    if (vgetq_lane_u32(cmp_result, 1) == 0)
      return false;
    if (vgetq_lane_u32(cmp_result, 2) == 0)
      return false;
    if (vgetq_lane_u32(cmp_result, 3) == 0)
      return false;
  }

  // Postamble loop
  for (int v = postamble_start; v < v_size; ++v)
  {
    if (vector[v] != 0.0)
      return false;
  }
  return true;
}

} // namespace ruy
} // namespace nnfw

#endif // USE_NEON

#endif // __NNFW_RUY_NEON_TENSOR_UTILS_H__
