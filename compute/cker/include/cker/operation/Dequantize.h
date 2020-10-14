/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __NNFW_CKER_DEQUANTIZE_H__
#define __NNFW_CKER_DEQUANTIZE_H__

#include "cker/Shape.h"
#include "cker/Types.h"
#include "cker/neon/neon_check.h"

namespace nnfw
{
namespace cker
{

#ifdef USE_NEON
namespace
{
inline void ScaleWithNewZeroPoint(const int32x4_t input, const float32x4_t scale_dup,
                                  const float32x4_t zero_times_scale_dup, float32x4_t *output)
{
#ifdef __ARM_FEATURE_FMA
  *output = vfmaq_f32(zero_times_scale_dup, vcvtq_f32_s32(input), scale_dup);
#else
  *output = vaddq_f32(vmulq_f32(vcvtq_f32_s32(input), scale_dup), zero_times_scale_dup);
#endif
}
} // namespace
#endif // USE_NEON

inline void Dequantize(const Shape &input_shape, const uint8_t *input_data,
                       const Shape &output_shape, float *output_data, const float scale,
                       const int32_t zero_point)
{
  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  int i = 0;
#ifdef USE_NEON
  const float32x4_t scale_dup = vdupq_n_f32(static_cast<float>(scale));
  const float32x4_t zero_times_scale_dup = vdupq_n_f32(static_cast<float>(-zero_point * scale));
  for (; i <= flat_size - 8; i += 8)
  {
    const uint8x8_t input_u8 = vld1_u8(input_data + i);
    const uint16x8_t input_u16 = vmovl_u8(input_u8);
    const int16x8_t input_s16 = vreinterpretq_s16_u16(input_u16);
    const int16x4_t input_s16_low = vget_low_s16(input_s16);
    const int16x4_t input_s16_high = vget_high_s16(input_s16);
    const int32x4_t val_low = vmovl_s16(input_s16_low);
    const int32x4_t val_high = vmovl_s16(input_s16_high);

    float32x4_t result_low, result_high;
    ScaleWithNewZeroPoint(val_low, scale_dup, zero_times_scale_dup, &result_low);
    ScaleWithNewZeroPoint(val_high, scale_dup, zero_times_scale_dup, &result_high);

    vst1q_f32(output_data + i, result_low);
    vst1q_f32(output_data + i + 4, result_high);
  }
#endif // NEON
  for (; i < flat_size; ++i)
  {
    const int32_t val = input_data[i];
    const float result = static_cast<float>(scale * (val - zero_point));
    output_data[i] = result;
  }
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_DEQUANTIZE_H__
