/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
 * Copyright (c) 2018 David Rowe
 *               2018 Mozilla
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

#ifndef __NNFW_CKER_EXP_H__
#define __NNFW_CKER_EXP_H__

#include "cker/Shape.h"
#if defined(USE_FAST_SOFTMAX)
#include "cker/neon/neon_check.h"
#endif

#include <cmath>

namespace nnfw
{
namespace cker
{

#if defined(USE_FAST_SOFTMAX) && defined(USE_NEON)
inline float32x4_t exp4_approx(float32x4_t x)
{
  int32x4_t i;
  float32x4_t xf;

  x = vmaxq_f32(vminq_f32(x, vdupq_n_f32(88.f)), vdupq_n_f32(-88.f));

  /* express exp(x) as exp2(x/log(2)), add 127 for the exponent later */
  x = vmlaq_f32(vdupq_n_f32(127.f), x, vdupq_n_f32(1.44269504f));

  /* split into integer and fractional parts */
  i = vcvtq_s32_f32(x);
  xf = vcvtq_f32_s32(i);
  x = vsubq_f32(x, xf);

  float32x4_t K0 = vdupq_n_f32(0.99992522f);
  float32x4_t K1 = vdupq_n_f32(0.69583354f);
  float32x4_t K2 = vdupq_n_f32(0.22606716f);
  float32x4_t K3 = vdupq_n_f32(0.078024523f);
  float32x4_t Y = vmlaq_f32(K0, x, vmlaq_f32(K1, x, vmlaq_f32(K2, K3, x)));

  /* compute 2^i */
  float32x4_t exponent = vreinterpretq_f32_s32(vshlq_n_s32(i, 23));

  Y = vmulq_f32(Y, exponent);
  return Y;
}

inline float celt_exp(float x)
{
  float out[4];
  float32x4_t X, Y;
  X = vdupq_n_f32(x);
  Y = exp4_approx(X);
  vst1q_f32(out, Y);
  return out[0];
}
#endif

inline void Exp(const Shape &input_shape, const float *input_data, const Shape &output_shape,
                float *output_data)
{
  const int size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < size; i++)
  {
    output_data[i] = std::exp(input_data[i]);
  }
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_EXP_H__
