/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __NNFW_CKER_ROUND_H__
#define __NNFW_CKER_ROUND_H__

#include "cker/Shape.h"
#include "cker/Utils.h"

#include <cmath>

namespace nnfw
{
namespace cker
{

inline float RoundToNearest(float value)
{
  auto floor_val = std::floor(value);
  auto diff = value - floor_val;
  if ((diff < 0.5f) || ((diff == 0.5f) && (static_cast<int>(floor_val) % 2 == 0)))
  {
    return floor_val;
  }
  else
  {
    return floor_val = floor_val + 1.0f;
  }
}

#ifdef USE_NEON

inline int32x4_t RoundToNearest(const float32x4_t input)
{
#if defined(__aarch64__) || defined(__SSSE3__)
  // Note: vcvtnq_s32_f32 is not available in ARMv7
  return vcvtnq_s32_f32(input);
#else
  static const float32x4_t zero_val_dup = vdupq_n_f32(0.0f);
  static const float32x4_t point5_val_dup = vdupq_n_f32(0.5f);
  static const float32x4_t minus_point5_val_dup = vdupq_n_f32(-0.5f);

  const uint32x4_t mask = vcltq_f32(input, zero_val_dup);
  const float32x4_t round = vbslq_f32(mask, minus_point5_val_dup, point5_val_dup);
  return vcvtq_s32_f32(vaddq_f32(input, round));
#endif // defined(__aarch64__) || defined(__SSSE3__)
}

#endif // NEON

inline void Round(const Shape &input_shape, const float *input_data, const Shape &output_shape,
                  float *output_data)
{
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < flat_size; ++i)
  {
    // Note that this implementation matches that of tensorFlow tf.round
    // and corresponds to the bankers rounding method.
    // cfenv (for fesetround) is not yet supported universally on Android, so
    // using a work around.
    output_data[i] = RoundToNearest(input_data[i]);
  }
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_ROUND_H__
