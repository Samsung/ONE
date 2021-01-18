/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2018 The TensorFlow Authors. All Rights Reserved.*
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

#ifndef __NNFW_CKER_QUANTIZE_H__
#define __NNFW_CKER_QUANTIZE_H__

#include "cker/Shape.h"
#include "cker/Types.h"
#include "cker/Utils.h"
#include <cassert>
#include <iostream>
#include <stdexcept>

namespace nnfw
{
namespace cker
{
template <typename InputT, typename OutputT>
inline void Quantize(const Shape &input_shape, const InputT *input_data, const Shape &output_shape,
                     OutputT *output_data, const float output_scale, const int32_t output_offset)
{
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  int min_val = std::numeric_limits<OutputT>::min();
  int max_val = std::numeric_limits<OutputT>::max();

  for (int i = 0; i < flat_size; i++)
  {
    int32_t unclamped = static_cast<int32_t>(round(input_data[i] / output_scale)) + output_offset;
    int32_t clamped = std::min(std::max(unclamped, min_val), max_val);
    output_data[i] = clamped;
  }
}

inline void Quantize(const int32_t *multiplier, const int32_t *shift, int32_t channel_size,
                     int32_t total_size, int32_t output_zp, int32_t output_min, int32_t output_max,
                     int32_t *scratch, int8_t *output)
{
  // Here we're trying to quantize the raw accumulators:
  //        output_channels
  //       data data data data data
  // rows  data data data data data
  //       data data data data data
  //          ....
  //
  // In order to minimize the reload of the multipliers & shifts, once we load
  // the multipliers & shifts, we load & quantize the raw accumulators for every
  // row.
#ifdef USE_NEON
  const int32x4_t output_offset_vec = vdupq_n_s32(output_zp);
  const int32x4_t output_activation_min_vec = vdupq_n_s32(output_min);
  const int32x4_t output_activation_max_vec = vdupq_n_s32(output_max);
  const int32x4_t zeros = vdupq_n_s32(0);
#endif

  assert(total_size % channel_size == 0);
  const int32_t rows = total_size / channel_size;

  int c = 0;

#ifdef USE_NEON
  using gemmlowp::RoundingDivideByPOT;
  for (; c <= channel_size - 8; c += 8)
  {
    int32x4_t out_shift_1 = vld1q_s32(shift + c);
    int32x4_t out_shift_2 = vld1q_s32(shift + c + 4);
    int32x4_t left_shift_1 = vmaxq_s32(out_shift_1, zeros);
    int32x4_t left_shift_2 = vmaxq_s32(out_shift_2, zeros);

    // Right shift will be performed as left shift with negative values.
    int32x4_t right_shift_1 = vminq_s32(out_shift_1, zeros);
    int32x4_t right_shift_2 = vminq_s32(out_shift_2, zeros);

    int32x4_t out_mul_1 = vld1q_s32(multiplier + c);
    int32x4_t out_mul_2 = vld1q_s32(multiplier + c + 4);
    for (int n = 0; n < rows; ++n)
    {
      int loc = n * channel_size + c;
      int32x4_t acc_1 = vld1q_s32(scratch + loc);
      int32x4_t acc_2 = vld1q_s32(scratch + loc + 4);

      // Saturating Rounding Doubling High Mul.
      acc_1 = vshlq_s32(acc_1, left_shift_1);
      acc_1 = vqrdmulhq_s32(acc_1, out_mul_1);
      acc_2 = vshlq_s32(acc_2, left_shift_2);
      acc_2 = vqrdmulhq_s32(acc_2, out_mul_2);

      // Rounding Dividing By POT.
      acc_1 = vrshlq_s32(acc_1, right_shift_1);
      acc_2 = vrshlq_s32(acc_2, right_shift_2);

      // Add the output offset.
      acc_1 = vaddq_s32(acc_1, output_offset_vec);
      acc_2 = vaddq_s32(acc_2, output_offset_vec);

      // Apply the activation function.
      acc_1 = vmaxq_s32(acc_1, output_activation_min_vec);
      acc_1 = vminq_s32(acc_1, output_activation_max_vec);
      acc_2 = vmaxq_s32(acc_2, output_activation_min_vec);
      acc_2 = vminq_s32(acc_2, output_activation_max_vec);

      // Saturating cast to int8 and store to destination.
      const int16x4_t acc_s16_1 = vqmovn_s32(acc_1);
      const int16x4_t acc_s16_2 = vqmovn_s32(acc_2);
      const int16x8_t res_s16 = vcombine_s16(acc_s16_1, acc_s16_2);
      const int8x8_t res_s8 = vqmovn_s16(res_s16);
      vst1_s8(output + loc, res_s8);
    }
  }

#endif // USE_NEON
  // Handle leftover values, one by one. This is very slow.
  for (; c < channel_size; c++)
  {
    for (int n = 0; n < rows; ++n)
    {
      int loc = n * channel_size + c;
      int32_t acc = scratch[loc];
      acc = MultiplyByQuantizedMultiplier(acc, multiplier[c], shift[c]);
      acc += output_zp;
      acc = std::max(acc, output_min);
      acc = std::min(acc, output_max);
      output[loc] = static_cast<int8_t>(acc);
    }
  }
}

template <typename input_type, typename output_type>
inline void Requantize(const input_type *input_data, int32_t size,
                       int32_t effective_scale_multiplier, int32_t effective_scale_shift,
                       int32_t input_zeropoint, int32_t output_zeropoint, output_type *output_data)
{
  assert(!"Requantize: not supported type. It shouldn't reach here.");
  UNUSED_ALL(input_data, size, effective_scale_multiplier, effective_scale_shift, input_zeropoint,
             output_zeropoint, output_data);
}

template <>
inline void Requantize<uint8_t, int8_t>(const uint8_t *input_data, int32_t size,
                                        int32_t effective_scale_multiplier,
                                        int32_t effective_scale_shift, int32_t input_zeropoint,
                                        int32_t output_zeropoint, int8_t *output_data)
{
  static constexpr int32_t kMinOutput = std::numeric_limits<int8_t>::min();
  static constexpr int32_t kMaxOutput = std::numeric_limits<int8_t>::max();

  int i = 0;
#ifdef USE_NEON
  // Constants.
  const int32x4_t input_zero_point_dup = vdupq_n_s32(-input_zeropoint);
  const int32x4_t output_zero_point_dup = vdupq_n_s32(output_zeropoint);
  const int32x4_t min_val_dup = vdupq_n_s32(kMinOutput);
  const int32x4_t max_val_dup = vdupq_n_s32(kMaxOutput);

  for (; i <= size - 16; i += 16)
  {
    const uint8x16_t input_vec = vld1q_u8(input_data + i);
    const uint16x8_t first_half = vmovl_u8(vget_low_u8(input_vec));
    const uint16x8_t second_half = vmovl_u8(vget_high_u8(input_vec));
    int32x4x4_t input;
    input.val[0] = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(first_half)));
    input.val[1] = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(first_half)));
    input.val[2] = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(second_half)));
    input.val[3] = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(second_half)));
    input.val[0] = vaddq_s32(input.val[0], input_zero_point_dup);
    input.val[1] = vaddq_s32(input.val[1], input_zero_point_dup);
    input.val[2] = vaddq_s32(input.val[2], input_zero_point_dup);
    input.val[3] = vaddq_s32(input.val[3], input_zero_point_dup);

    int32x4x4_t result =
      MultiplyByQuantizedMultiplier4Rows(input, effective_scale_multiplier, effective_scale_shift);

    result.val[0] = vaddq_s32(result.val[0], output_zero_point_dup);
    result.val[1] = vaddq_s32(result.val[1], output_zero_point_dup);
    result.val[2] = vaddq_s32(result.val[2], output_zero_point_dup);
    result.val[3] = vaddq_s32(result.val[3], output_zero_point_dup);
    result.val[0] = vmaxq_s32(vminq_s32(result.val[0], max_val_dup), min_val_dup);
    result.val[1] = vmaxq_s32(vminq_s32(result.val[1], max_val_dup), min_val_dup);
    result.val[2] = vmaxq_s32(vminq_s32(result.val[2], max_val_dup), min_val_dup);
    result.val[3] = vmaxq_s32(vminq_s32(result.val[3], max_val_dup), min_val_dup);

    const int16x4_t narrowed_val_1 = vqmovn_s32(result.val[0]);
    const int16x4_t narrowed_val_2 = vqmovn_s32(result.val[1]);
    const int16x4_t narrowed_val_3 = vqmovn_s32(result.val[2]);
    const int16x4_t narrowed_val_4 = vqmovn_s32(result.val[3]);
    const int16x8_t output_first_half = vcombine_s16(narrowed_val_1, narrowed_val_2);
    const int16x8_t output_second_half = vcombine_s16(narrowed_val_3, narrowed_val_4);
    const int8x8_t narrowed_first_half = vqmovn_s16(output_first_half);
    const int8x8_t narrowed_second_half = vqmovn_s16(output_second_half);
    const int8x16_t narrowed_result = vcombine_s8(narrowed_first_half, narrowed_second_half);
    vst1q_s8(output_data + i, narrowed_result);
  }

#endif
  for (; i < size; ++i)
  {
    const int32_t input = input_data[i] - input_zeropoint;
    const int32_t output =
      MultiplyByQuantizedMultiplier(input, effective_scale_multiplier, effective_scale_shift) +
      output_zeropoint;
    const int32_t clamped_output = std::max(std::min(output, kMaxOutput), kMinOutput);
    output_data[i] = static_cast<int8_t>(clamped_output);
  }
}

template <>
inline void Requantize<int8_t, uint8_t>(const int8_t *input_data, int32_t size,
                                        int32_t effective_scale_multiplier,
                                        int32_t effective_scale_shift, int32_t input_zeropoint,
                                        int32_t output_zeropoint, uint8_t *output_data)
{
  static constexpr int32_t kMinOutput = std::numeric_limits<uint8_t>::min();
  static constexpr int32_t kMaxOutput = std::numeric_limits<uint8_t>::max();

  int i = 0;
#ifdef USE_NEON
  // Constants.
  const int32x4_t input_zero_point_dup = vdupq_n_s32(-input_zeropoint);
  const int32x4_t output_zero_point_dup = vdupq_n_s32(output_zeropoint);
  const int32x4_t min_val_dup = vdupq_n_s32(kMinOutput);
  const int32x4_t max_val_dup = vdupq_n_s32(kMaxOutput);

  for (; i <= size - 16; i += 16)
  {
    const int8x16_t input_vec = vld1q_s8(input_data + i);
    const int16x8_t first_half = vmovl_s8(vget_low_s8(input_vec));
    const int16x8_t second_half = vmovl_s8(vget_high_s8(input_vec));
    int32x4x4_t input;
    input.val[0] = vmovl_s16(vget_low_s16(first_half));
    input.val[1] = vmovl_s16(vget_high_s16(first_half));
    input.val[2] = vmovl_s16(vget_low_s16(second_half));
    input.val[3] = vmovl_s16(vget_high_s16(second_half));
    input.val[0] = vaddq_s32(input.val[0], input_zero_point_dup);
    input.val[1] = vaddq_s32(input.val[1], input_zero_point_dup);
    input.val[2] = vaddq_s32(input.val[2], input_zero_point_dup);
    input.val[3] = vaddq_s32(input.val[3], input_zero_point_dup);

    int32x4x4_t result =
      MultiplyByQuantizedMultiplier4Rows(input, effective_scale_multiplier, effective_scale_shift);

    result.val[0] = vaddq_s32(result.val[0], output_zero_point_dup);
    result.val[1] = vaddq_s32(result.val[1], output_zero_point_dup);
    result.val[2] = vaddq_s32(result.val[2], output_zero_point_dup);
    result.val[3] = vaddq_s32(result.val[3], output_zero_point_dup);
    result.val[0] = vmaxq_s32(vminq_s32(result.val[0], max_val_dup), min_val_dup);
    result.val[1] = vmaxq_s32(vminq_s32(result.val[1], max_val_dup), min_val_dup);
    result.val[2] = vmaxq_s32(vminq_s32(result.val[2], max_val_dup), min_val_dup);
    result.val[3] = vmaxq_s32(vminq_s32(result.val[3], max_val_dup), min_val_dup);

    const uint32x4_t result_val_1_unsigned = vreinterpretq_u32_s32(result.val[0]);
    const uint32x4_t result_val_2_unsigned = vreinterpretq_u32_s32(result.val[1]);
    const uint32x4_t result_val_3_unsigned = vreinterpretq_u32_s32(result.val[2]);
    const uint32x4_t result_val_4_unsigned = vreinterpretq_u32_s32(result.val[3]);

    const uint16x4_t narrowed_val_1 = vqmovn_u32(result_val_1_unsigned);
    const uint16x4_t narrowed_val_2 = vqmovn_u32(result_val_2_unsigned);
    const uint16x4_t narrowed_val_3 = vqmovn_u32(result_val_3_unsigned);
    const uint16x4_t narrowed_val_4 = vqmovn_u32(result_val_4_unsigned);
    const uint16x8_t output_first_half = vcombine_u16(narrowed_val_1, narrowed_val_2);
    const uint16x8_t output_second_half = vcombine_u16(narrowed_val_3, narrowed_val_4);
    const uint8x8_t narrowed_first_half = vqmovn_u16(output_first_half);
    const uint8x8_t narrowed_second_half = vqmovn_u16(output_second_half);
    const uint8x16_t narrowed_result = vcombine_u8(narrowed_first_half, narrowed_second_half);
    vst1q_u8(output_data + i, narrowed_result);
  }

#endif
  for (; i < size; ++i)
  {
    const int32_t input = input_data[i] - input_zeropoint;
    const int32_t output =
      MultiplyByQuantizedMultiplier(input, effective_scale_multiplier, effective_scale_shift) +
      output_zeropoint;
    const int32_t clamped_output = std::max(std::min(output, kMaxOutput), kMinOutput);
    output_data[i] = static_cast<uint8_t>(clamped_output);
  }
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_QUANTIZE_H__
