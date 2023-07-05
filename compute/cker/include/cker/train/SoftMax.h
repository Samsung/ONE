/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __NNFW_CKER_TRAIN_SOFTMAX_H__
#define __NNFW_CKER_TRAIN_SOFTMAX_H__

#include "cker/Shape.h"
#include "cker/Utils.h"
#include "cker/Types.h"
#include "cker/eigen/Utils.h"

#if __aarch64__ && __clang__
#define TFLITE_SOFTMAX_USE_UINT16_LUT
#endif

#include <Eigen/Core>
#include <fixedpoint/fixedpoint.h>
#include <cmath>

namespace nnfw
{
namespace cker
{
namespace train
{

namespace reference
{

// Note. This Softmax function supports all of dimensions
inline void Softmax(const SoftmaxParams &params, const Shape &input_shape, const float *input_data,
                    const Shape &output_shape, float *output_data)
{
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size = MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth = MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

  for (int i = 0; i < outer_size; ++i)
  {
    // Find max element value which we'll use to ensure numerical stability
    // taking advantage of the following equality:
    // exp(x[i])/sum(exp(x[i])) == exp(x[i]+C)/sum(exp(x[i]+C))
    float max = std::numeric_limits<float>::lowest();
    for (int c = 0; c < depth; ++c)
    {
      max = std::max(max, input_data[i * depth + c]);
    }

    // Compute sum.
    float sum = 0.f;
    for (int c = 0; c < depth; ++c)
    {
      sum += std::exp((input_data[i * depth + c] - max) * static_cast<float>(params.beta));
    }

    // Compute result.
    for (int c = 0; c < depth; ++c)
    {
      output_data[i * depth + c] =
        std::exp((input_data[i * depth + c] - max) * static_cast<float>(params.beta)) / sum;
    }
  }
}
} // namespace reference

// Performs softmax along the input of size (input_size * batch_size).
inline void Softmax(const float *in, const int input_size, const int batch_size, const float beta,
                    float *out)
{
  assert(input_size > 0);

  // For each batch
  for (int b = 0; b < batch_size; b++)
  {
    // Find the max coeff.
    float max_coeff = in[0];
    for (int i = 1; i < input_size; i++)
    {
      if (in[i] > max_coeff)
        max_coeff = in[i];
    }

    // Compute the normalized sum of exps.
    float exp_sum = 0.0;
    for (int i = 0; i < input_size; i++)
    {
      out[i] = std::exp((in[i] - max_coeff) * beta);
      exp_sum += out[i];
    }

    // Divide by the sum of exps.
    float reciprocal_sum_exp = 1.f / exp_sum;
    for (int i = 0; i < input_size; i++)
    {
      out[i] *= reciprocal_sum_exp;
    }

    // Advance in and out pointers for the next batch.
    in += input_size;
    out += input_size;
  }
}

inline void Softmax(const SoftmaxParams &params, const Shape &input_shape, const float *input_data,
                    const Shape &output_shape, float *output_data)
{
  // Validate whether if shapes of input and output are the same
  MatchingFlatSize(input_shape, output_shape);

  const auto in_mat = MapAsMatrixWithLastDimAsRows(input_data, input_shape);
  auto out_mat = MapAsMatrixWithLastDimAsRows(output_data, output_shape);
  // Compute the exponential first, removing the max coefficient for numerical
  // stability.
  out_mat = (in_mat.rowwise() - in_mat.colwise().maxCoeff()).array() * params.beta;
  // We are separating out the exp function so that exp can be vectorized.
  out_mat = out_mat.array().exp();
  // Normalize to get the activations.
  Eigen::Array<float, 1, Eigen::Dynamic> scale = out_mat.array().colwise().sum().inverse();
  out_mat.array().rowwise() *= scale;
}

template <typename T> inline int32_t QuantizeSoftmaxOutput(float prob_rescaled, int32_t zero_point)
{
  const int32_t prob_rnd = static_cast<int32_t>(std::round(prob_rescaled));
  return prob_rnd + zero_point;
}

#if !__aarch64__
// With ARM64, rounding is faster than add + truncation.
template <> inline int32_t QuantizeSoftmaxOutput<uint8_t>(float prob_rescaled, int32_t)
{
  return static_cast<int32_t>(prob_rescaled + 0.5f);
}
#endif

inline void PopulateSoftmaxLookupTable(float *table, float input_scale, float beta)
{
  const float scale = -input_scale * beta;
  const int32_t max_uint8 = std::numeric_limits<uint8_t>::max();
  for (int32_t val = 0; val <= max_uint8; ++val)
  {
    table[max_uint8 - val] = expf(scale * val);
  }
}

template <typename In, typename Out>
inline void Softmax(const SoftmaxParams &params, const Shape &input_shape, const In *input_data,
                    const Shape &output_shape, Out *output_data)
{
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int excluding_last_dim = MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int last_dim = MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

  const int32_t clamp_max = std::numeric_limits<Out>::max();
  const int32_t clamp_min = std::numeric_limits<Out>::min();
  for (int i = 0; i < excluding_last_dim; ++i)
  {
    int32_t max_val = std::numeric_limits<In>::min();
    // Find max quantized value.
    for (int j = 0; j < last_dim; ++j)
    {
      max_val = std::max(max_val, static_cast<int32_t>(input_data[j]));
    }

    float sum_exp = 0.0f;
    const int32_t max_uint8 = std::numeric_limits<uint8_t>::max();
    const float *table_offset = &params.table[max_uint8 - max_val];
    // Calculate normalizer sum(exp(x)).
    for (int j = 0; j < last_dim; ++j)
    {
      sum_exp += table_offset[input_data[j]];
    }

    const float inv_sum_exp = 1.0f / (sum_exp * params.scale);
    // Normalize and quantize probabilities.
    for (int j = 0; j < last_dim; ++j)
    {
      const float prob_rescaled = table_offset[input_data[j]] * inv_sum_exp;
      const int32_t prob_quantized = QuantizeSoftmaxOutput<Out>(prob_rescaled, params.zero_point);
      output_data[j] = static_cast<Out>(std::max(std::min(clamp_max, prob_quantized), clamp_min));
    }
    input_data += last_dim;
    output_data += last_dim;
  }
}

#ifdef TFLITE_SOFTMAX_USE_UINT16_LUT
// Looks up each element of <indices> in <table>, returns them in a vector.
inline uint8x16_t aarch64_lookup_vector(const uint8x16x4_t table[4], uint8x16_t indices)
{
  // Look up in 1st quarter of the table: top 2 bits of indices == 00
  uint8x16_t output1 = vqtbl4q_u8(table[0], indices);
  // Look up in 2nd quarter of the table: top 2 bits of indices == 01
  uint8x16_t output2 = vqtbl4q_u8(table[1], veorq_u8(indices, vdupq_n_u8(0x40)));
  // Look up in 3rd quarter of the table: top 2 bits of indices == 10
  uint8x16_t output3 = vqtbl4q_u8(table[2], veorq_u8(indices, vdupq_n_u8(0x80)));
  // Look up in 4th quarter of the table: top 2 bits of indices == 11
  uint8x16_t output4 = vqtbl4q_u8(table[3], veorq_u8(indices, vdupq_n_u8(0xc0)));

  // Combine result of the 4 lookups.
  return vorrq_u8(vorrq_u8(output1, output2), vorrq_u8(output3, output4));
}

inline void PopulateSoftmaxUInt8LookupTable(uint8_t *uint8_table1, uint8_t *uint8_table2,
                                            float input_scale, float beta)
{
  const float scale = input_scale * beta;
  const int32_t max_uint8 = std::numeric_limits<uint8_t>::max();
  const int32_t max_uint16 = std::numeric_limits<uint16_t>::max();

  for (int32_t val = 0; val <= max_uint8; ++val)
  {
    float input_to_exp = scale * (val - max_uint8);
    int32_t temp = static_cast<int>(expf(input_to_exp) * max_uint16 + 0.5);
    temp = std::min(max_uint16, temp);
    uint8_t part1 = temp >> 8;
    uint8_t part2 = temp & 0xff;
    uint8_table1[val] = static_cast<uint8_t>(part1);
    uint8_table2[val] = static_cast<uint8_t>(part2);
  }
}

inline int FindMaxValue(int size, const uint8_t *input_data, uint8_t offset)
{
  int32_t max_val = std::numeric_limits<uint8_t>::min();
  int j = 0;

  uint8x16_t max_val_dup = vdupq_n_u8(max_val);
  uint8x16_t offset_dup = vdupq_n_u8(offset);
  for (; j <= size - 16; j += 16)
  {
    uint8x16_t input_value = vld1q_u8(input_data + j);
    input_value = veorq_u8(input_value, offset_dup);
    max_val_dup = vmaxq_u8(input_value, max_val_dup);
  }
  max_val = std::max(max_val, static_cast<int32_t>(vmaxvq_u8(max_val_dup)));

  for (; j < size; ++j)
  {
    max_val = std::max(max_val, static_cast<int32_t>(input_data[j] ^ offset));
  }
  return max_val;
}

#ifdef USE_NEON
// Value_to_store layout:
// [high_high, high_low, low_high, low_low].
inline void StoreValue(int32x4x4_t value_to_store, int8_t *output)
{
  const int16x8_t result_1 =
    vcombine_s16(vqmovn_s32(value_to_store.val[1]), vqmovn_s32(value_to_store.val[0]));
  const int16x8_t result_2 =
    vcombine_s16(vqmovn_s32(value_to_store.val[3]), vqmovn_s32(value_to_store.val[2]));
  const int8x16_t result = vcombine_s8(vqmovn_s16(result_2), vqmovn_s16(result_1));
  vst1q_s8(output, result);
}

// Value_to_store layout:
// [high_high, high_low, low_high, low_low].
inline void StoreValue(int32x4x4_t value_to_store, uint8_t *output)
{
  const uint16x8_t result_1 =
    vcombine_u16(vqmovn_u32(vreinterpretq_u32_s32(value_to_store.val[1])),
                 vqmovn_u32(vreinterpretq_u32_s32(value_to_store.val[0])));
  const uint16x8_t result_2 =
    vcombine_u16(vqmovn_u32(vreinterpretq_u32_s32(value_to_store.val[3])),
                 vqmovn_u32(vreinterpretq_u32_s32(value_to_store.val[2])));
  const uint8x16_t result = vcombine_u8(vqmovn_u16(result_2), vqmovn_u16(result_1));
  vst1q_u8(output, result);
}

#endif

template <typename In, typename Out>
inline void SoftmaxInt8LUT(const SoftmaxParams &params, const Shape &input_shape,
                           const In *input_data, const Shape &output_shape, Out *output_data)
{
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int excluding_last_dim = MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int last_dim = MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

  const int32_t clamp_max = std::numeric_limits<Out>::max();
  const int32_t clamp_min = std::numeric_limits<Out>::min();

  // Offset is used to interpret the input data "correctly".
  // If the input is uint8, the data will be unchanged.
  // If the input is int8, since it will be reinterpret as uint8.
  // e.g.,
  // int8 127 will be applied "offset" to become 255 in uint8.
  uint8_t offset = 0;
  if (std::is_same<In, int8_t>::value)
  {
    offset = 0x80;
  }

  const uint8_t *input_data_uint = reinterpret_cast<const uint8_t *>(input_data);

  // This code uses ARM64-only instructions.
  // TODO(b/143709993): Port to ARMv7

  // Load the tables into registers. (4*4 128-bit registers)
  uint8x16x4_t table1[4];
  table1[0] = vld1q_u8_x4(params.uint8_table1 + 16 * 4 * 0);
  table1[1] = vld1q_u8_x4(params.uint8_table1 + 16 * 4 * 1);
  table1[2] = vld1q_u8_x4(params.uint8_table1 + 16 * 4 * 2);
  table1[3] = vld1q_u8_x4(params.uint8_table1 + 16 * 4 * 3);

  uint8x16x4_t table2[4];
  table2[0] = vld1q_u8_x4(params.uint8_table2 + 16 * 4 * 0);
  table2[1] = vld1q_u8_x4(params.uint8_table2 + 16 * 4 * 1);
  table2[2] = vld1q_u8_x4(params.uint8_table2 + 16 * 4 * 2);
  table2[3] = vld1q_u8_x4(params.uint8_table2 + 16 * 4 * 3);

  for (int i = 0; i < excluding_last_dim; ++i)
  {
    // Find max quantized value.
    int32_t max_val = FindMaxValue(last_dim, input_data_uint, offset);

    int32_t sum_exp = 0;
    const int32_t max_uint8 = std::numeric_limits<uint8_t>::max();
    const uint8_t table_offset = max_uint8 - max_val;

    // Calculate normalizer sum(exp(x)).
    int sum_j = 0;
    uint8x16_t table_offset_dup = vdupq_n_u8(table_offset);
    uint8x16_t offset_dup = vdupq_n_u8(offset);
    uint32x4_t sum_4 = vdupq_n_u32(0);
    const int multiplier_shift = 8;
    for (; sum_j <= last_dim - 16; sum_j += 16)
    {
      uint8x16_t input_value = vld1q_u8(input_data_uint + sum_j);
      input_value = veorq_u8(input_value, offset_dup);
      input_value = vaddq_u8(input_value, table_offset_dup);

      const uint8x16_t output1 = aarch64_lookup_vector(table1, input_value);
      const uint8x16_t output2 = aarch64_lookup_vector(table2, input_value);

      uint16x8_t exp_value1 = vshll_n_u8(vget_high_u8(output1), multiplier_shift);
      uint16x8_t exp_value2 = vshll_n_u8(vget_low_u8(output1), multiplier_shift);

      exp_value1 = vaddw_u8(exp_value1, vget_high_u8(output2));
      exp_value2 = vaddw_u8(exp_value2, vget_low_u8(output2));

      sum_4 = vpadalq_u16(sum_4, exp_value1);
      sum_4 = vpadalq_u16(sum_4, exp_value2);
    }
    int temp = vgetq_lane_u32(sum_4, 0) + vgetq_lane_u32(sum_4, 1) + vgetq_lane_u32(sum_4, 2) +
               vgetq_lane_u32(sum_4, 3);
    sum_exp += temp;

    for (; sum_j < last_dim; ++sum_j)
    {
      const uint8_t index = (input_data_uint[sum_j] ^ offset) + table_offset;

      uint8_t part1 = params.uint8_table1[index];
      uint8_t part2 = params.uint8_table2[index];
      sum_exp += ((part1 << 8) + part2);
    }

    const float inv_sum_exp = 1.0f / (sum_exp * params.scale);

    int32_t multiplier, shift;
    QuantizeMultiplier(inv_sum_exp, &multiplier, &shift);

    // Normalize and quantize probabilities.
    int j = 0;
    const int32x4_t output_zp_dup = vdupq_n_s32(params.zero_point);
    const int32x4_t max_val_dup = vdupq_n_s32(clamp_max);
    const int32x4_t min_val_dup = vdupq_n_s32(clamp_min);

    for (; j <= last_dim - 16; j += 16)
    {
      uint8x16_t input_value = vld1q_u8(input_data_uint + j);
      input_value = veorq_u8(input_value, offset_dup);
      input_value = vaddq_u8(input_value, table_offset_dup);

      const uint8x16_t output1 = aarch64_lookup_vector(table1, input_value);
      const uint8x16_t output2 = aarch64_lookup_vector(table2, input_value);

      uint16x8_t exp_value1 = vshll_n_u8(vget_high_u8(output1), multiplier_shift);
      uint16x8_t exp_value2 = vshll_n_u8(vget_low_u8(output1), multiplier_shift);

      exp_value1 = vaddw_u8(exp_value1, vget_high_u8(output2));
      exp_value2 = vaddw_u8(exp_value2, vget_low_u8(output2));

      int32x4x4_t output_value;
      output_value.val[0] = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(exp_value1)));
      output_value.val[1] = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(exp_value1)));
      output_value.val[2] = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(exp_value2)));
      output_value.val[3] = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(exp_value2)));

      int32x4x4_t temp_val = MultiplyByQuantizedMultiplier4Rows(output_value, multiplier, shift);

      temp_val.val[0] = vaddq_s32(temp_val.val[0], output_zp_dup);
      temp_val.val[1] = vaddq_s32(temp_val.val[1], output_zp_dup);
      temp_val.val[2] = vaddq_s32(temp_val.val[2], output_zp_dup);
      temp_val.val[3] = vaddq_s32(temp_val.val[3], output_zp_dup);

      temp_val.val[0] = vmaxq_s32(vminq_s32(temp_val.val[0], max_val_dup), min_val_dup);
      temp_val.val[1] = vmaxq_s32(vminq_s32(temp_val.val[1], max_val_dup), min_val_dup);
      temp_val.val[2] = vmaxq_s32(vminq_s32(temp_val.val[2], max_val_dup), min_val_dup);
      temp_val.val[3] = vmaxq_s32(vminq_s32(temp_val.val[3], max_val_dup), min_val_dup);

      StoreValue(temp_val, output_data + j);
    }
    for (; j < last_dim; ++j)
    {
      const uint8_t index = (input_data_uint[j] ^ offset) + table_offset;
      const uint8_t part1 = params.uint8_table1[index];
      const uint8_t part2 = params.uint8_table2[index];
      const int32_t exp_value = (part1 << 8) + part2;
      const int32_t output_value = MultiplyByQuantizedMultiplier(exp_value, multiplier, shift);

      output_data[j] = static_cast<Out>(
        std::max(std::min(clamp_max, output_value + params.zero_point), clamp_min));
    }
    input_data_uint += last_dim;
    output_data += last_dim;
  }
}
#endif

} // namespace train
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_TRAIN_SOFTMAX_H__
