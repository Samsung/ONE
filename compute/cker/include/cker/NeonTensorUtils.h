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

#ifndef __NNFW_CKER_NEON_TENSOR_UTILS_H__
#define __NNFW_CKER_NEON_TENSOR_UTILS_H__

#include <ruy/path.h>
#include <ruy/ruy.h>
#include <ruy/detect_arm.h>
#include "cker/Types.h"
#include "cker/neon/neon_check.h"
#include "cker/ruy/RuySupport.h"
#include "util/logging.h"

#include <cassert>
#include <cmath>

#ifdef USE_NEON

#define kFloatWeightsPerNeonLane 4

namespace nnfw
{
namespace cker
{

namespace
{

// Allocates, at least, size bytes of uninitialized storage whose alignment is
// specified by alignment. The size parameter must be an integral multiple of
// alignment.
// Caller is responsible by freeing the allocated memory by calling free on
// the passed freeing_buffer pointer.
void *aligned_alloc(size_t alignment, size_t size, void **freeing_buffer)
{
  *freeing_buffer = malloc(size + alignment);
  const size_t offset = ((uintptr_t)*freeing_buffer) % alignment;                          // NOLINT
  return offset == 0 ? *freeing_buffer : ((char *)*freeing_buffer + (alignment - offset)); // NOLINT
}

} // namespace

bool NeonIsZeroVector(const float *vector, int v_size)
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

void NeonCpuBackendGemm(const int8_t *input, const int32_t *bias,
                        const int8_t *input_to_gate_weights, int32_t n_batch, int32_t n_input,
                        int32_t n_output, int32_t, int32_t *scratch)
{
  MatrixParams<int8_t> lhs_params;
  lhs_params.order = Order::kRowMajor;
  lhs_params.rows = n_output;
  lhs_params.cols = n_input;
  lhs_params.cacheable = true;

  MatrixParams<int8_t> rhs_params;
  rhs_params.order = Order::kColMajor;
  rhs_params.rows = n_input;
  rhs_params.cols = n_batch;

  MatrixParams<int32_t> dst_params;
  dst_params.order = Order::kColMajor;
  dst_params.rows = n_output;
  dst_params.cols = n_batch;

  GemmParams<int32_t, int32_t> gemm_params;
  if (bias)
  {
    gemm_params.bias = bias;
  }

  // Below code is from tflite::cpu_backend_gemm::detail::GemmImplUsingRuy
  ruy::Context *ruy_context = ruy_support::GetRuyContext();

  ruy::Matrix<int8_t> ruy_lhs;
  ruy::Matrix<int8_t> ruy_rhs;
  ruy::Matrix<int32_t> ruy_dst;
  ruy_support::MakeRuyMatrix(lhs_params, input_to_gate_weights, &ruy_lhs);
  ruy_support::MakeRuyMatrix(rhs_params, input, &ruy_rhs);
  ruy_support::MakeRuyMatrix(dst_params, scratch, &ruy_dst);

  ruy::BasicSpec<int32_t, int32_t> ruy_spec;
  ruy_support::MakeRuySpec(gemm_params, &ruy_spec);

  constexpr ruy::Path kRuyPath = ruy::kAllPaths;
  ruy::Mul<kRuyPath>(ruy_lhs, ruy_rhs, ruy_spec, ruy_context, &ruy_dst);
}

void NeonSymmetricQuantizeFloats(const float *values, const int size, int8_t *quantized_values,
                                 float *min, float *max, float *scaling_factor)
{
  // TODO(raziel): vectorize min/max calculation.
  auto minmax = std::minmax_element(values, values + size);
  *min = *minmax.first;
  *max = *minmax.second;
  const int kScale = 127;
  const float range = std::max(std::abs(*min), std::abs(*max));
  if (range == 0)
  {
    memset(quantized_values, 0, size * sizeof(int8_t));
    *scaling_factor = 1;
    return;
  }
  *scaling_factor = range / kScale;
  const float scaling_factor_inv = kScale / range;

  const int postamble_start = size - (size & (2 * kFloatWeightsPerNeonLane - 1));

  // Vectorized constants.
  const float32x4_t q_factor_f32x4 = vmovq_n_f32(scaling_factor_inv);
  const float32x4_t point5_f32x4 = vmovq_n_f32(0.5);
  const float32x4_t zero_f32x4 = vmovq_n_f32(0.0);
  const int32x4_t scale_i32x4 = vmovq_n_s32(kScale);
  const int32x4_t neg_scale_i32x4 = vmovq_n_s32(-kScale);

  for (int i = 0; i < postamble_start; i += 2 * kFloatWeightsPerNeonLane)
  {
    // Implements the vectorized version of the following:
    // const int32_t quantized_value = static_cast<int32>(
    //    std::round(*scaling_factor * values[i]));
    // Since the vectorized round intrinsics (vrndqa_f32) is not supported
    // on all Neon flavors, we use the following method for rounding: if (x
    // < 0) (int)(x - 0.5) if (x >= 0) (int)(x + 0.5)
    float32x4_t value0_f32x4 = vld1q_f32(&values[i]);
    float32x4_t value1_f32x4 = vld1q_f32(&values[i + kFloatWeightsPerNeonLane]);
    float32x4_t mul0_f32x4 = vmulq_f32(value0_f32x4, q_factor_f32x4);
    float32x4_t mul1_f32x4 = vmulq_f32(value1_f32x4, q_factor_f32x4);

    int32x4_t cmp_with_zero0_ui32x4 = (int32x4_t)vcltq_f32(mul0_f32x4, zero_f32x4); // NOLINT
    int32x4_t cmp_with_zero1_ui32x4 = (int32x4_t)vcltq_f32(mul1_f32x4, zero_f32x4); // NOLINT

    float32x4_t cmp_with_zero0_f32x4 = vcvtq_f32_s32(cmp_with_zero0_ui32x4);
    float32x4_t cmp_with_zero1_f32x4 = vcvtq_f32_s32(cmp_with_zero1_ui32x4);
    cmp_with_zero0_f32x4 = vaddq_f32(cmp_with_zero0_f32x4, point5_f32x4);
    cmp_with_zero1_f32x4 = vaddq_f32(cmp_with_zero1_f32x4, point5_f32x4);

    mul0_f32x4 = vaddq_f32(mul0_f32x4, cmp_with_zero0_f32x4);
    mul1_f32x4 = vaddq_f32(mul1_f32x4, cmp_with_zero1_f32x4);

    int32x4_t f2i0_i32x4 = vcvtq_s32_f32(mul0_f32x4);
    int32x4_t f2i1_i32x4 = vcvtq_s32_f32(mul1_f32x4);

    // Implements the vectorized version of the folowing block:
    //  quantized_values[i] = std::min(kScale, std::max(-kScale,
    //  quantized_value));
    int32x4_t max0_i32x4 = vmaxq_s32(f2i0_i32x4, neg_scale_i32x4);
    int32x4_t max1_i32x4 = vmaxq_s32(f2i1_i32x4, neg_scale_i32x4);
    int32x4_t min0_i32x4 = vminq_s32(max0_i32x4, scale_i32x4);
    int32x4_t min1_i32x4 = vminq_s32(max1_i32x4, scale_i32x4);

    int16x4_t min0_16x4 = vmovn_s32(min0_i32x4);
    int16x4_t min1_16x4 = vmovn_s32(min1_i32x4);

    int16x8_t min_16x8 = vcombine_s16(min0_16x4, min1_16x4);
    int8x8_t min_s8x8 = vqmovn_s16(min_16x8);
    vst1_s8(&quantized_values[i], min_s8x8);
  }

  for (int i = postamble_start; i < size; ++i)
  {
    const int32_t quantized_value =
        static_cast<int32_t>(std::round(scaling_factor_inv * values[i]));
    quantized_values[i] = std::min(kScale, std::max(-kScale, quantized_value));
  }
}

void NeonMatrixBatchVectorMultiplyAccumulate(const int8_t *__restrict__ matrix, const int m_rows,
                                             const int m_cols, const int8_t *__restrict__ vectors,
                                             const float *scaling_factors, int n_batch,
                                             float *__restrict__ result, int result_stride)
{
  const int kWeightsPerUint32 = 4;
  const int kWeightsPerNeonLane = 16;
  // If the number of rows is not divisible by kWeightsPerUint32, we set a
  // flag and allocate an aligned memory block. The flag is used to use the
  // aligned memory block later in the kernel loop.
  bool unaligned = false;
  int8_t *aligned_row = nullptr;
  void *aligned_row_free = nullptr;
  if ((m_cols & (kWeightsPerUint32 - 1)) != 0)
  {
    unaligned = true;
    aligned_row = (int8_t *)aligned_alloc(kWeightsPerUint32, m_cols, // NOLINT
                                          &aligned_row_free);
  }
  void *aligned_vec_free = nullptr;
  int8_t *aligned_vec = (int8_t *)aligned_alloc(kWeightsPerUint32, m_cols, // NOLINT
                                                &aligned_vec_free);

  // If m_cols is not at least kWeightsPerNeonLane, we cannot use the main
  // vectorized loop, and we need to process sequentially. postamble_start shows
  // the start index where this should happen.
  const int postamble_start = m_cols - (m_cols & (kWeightsPerNeonLane - 1));

  int batch, row, col;
  for (batch = 0; batch < n_batch; ++batch)
  {
    const float batch_scaling_factor = scaling_factors[batch];
    // Copy the vector data to an aligned vector.
    memcpy(aligned_vec, vectors + batch * m_cols, sizeof(int8_t) * m_cols);
    // Compute dot-product for every column.
    for (row = 0; row < m_rows; ++row, result += result_stride)
    {
      // Get the address of the first element of the row.
      int8_t *row_ptr = (int8_t *)matrix + row * m_cols; // NOLINT
      if (unaligned)
      {
        memcpy(aligned_row, row_ptr, sizeof(int8_t) * m_cols);
        row_ptr = aligned_row;
      }

      // Initialize the dot product sum for the row to 0.
      int32x4_t dotprod = vmovq_n_s32(0);

      // Prefetch the row to cache.
      __builtin_prefetch(row_ptr, 0 /* prefetch for read */, 3 /* temporal locality */);

      // For every block of 16 8-bit elements.
      col = 0;
      for (; col < postamble_start; col += kWeightsPerNeonLane)
      {
        // Load 16 8-bit values from the row and vector, each, to operate on.
        // Here the assumption is that each buffer is 4-byte aligned.
        assert(((uintptr_t)(&row_ptr[col]) & (kWeightsPerUint32 - 1)) == 0);
        const int8x16_t s1_8x16 = vld1q_s8((const int8_t *)(aligned_vec + col));
        const int8x16_t s2_8x16 = vld1q_s8((const int8_t *)(row_ptr + col));
        // Multiply the low bits (i.e. the lower 8 8bit numbers in the
        // registers).
        int16x8_t prod_16x8 = vmull_s8(vget_low_s8(s1_8x16), vget_low_s8(s2_8x16));
        // Multiply the high bits (i.e. the lower 8 8bit numbers in the
        // registers), and accumulate with the result of the low bits product.
        // The assumption here is that overflow will not happen as we quantize
        // our values to be in the range [-127, 127]. As such the sum of the 2
        // products is always strictly smaller than 15-bits (32767 in absolute
        // value).
        prod_16x8 = vmlal_s8(prod_16x8, vget_high_s8(s1_8x16), vget_high_s8(s2_8x16));

        dotprod = vpadalq_s16(dotprod, prod_16x8);
      } // for col

      int32_t postable_sum = 0;
      // Postamble loop.
      // TODO(raziel): if (ABSL_PREDICT_FALSE(postamble_start < m_rows))
      if (postamble_start < m_cols)
      {
        col = postamble_start;
        if ((m_cols - postamble_start) >= (kWeightsPerNeonLane >> 1))
        {
          // Load 8 8-bit values from the row and column each to operate on.
          // Here the assumption is that each buffer is 4-bytes aligned.
          assert(((uintptr_t)(&row_ptr[col]) & (kWeightsPerUint32 - 1)) == 0);
          const int8x8_t s1_8x8 = vld1_s8((const int8_t *)(aligned_vec + col));
          const int8x8_t s2_8x8 = vld1_s8((const int8_t *)(row_ptr + col));
          const int16x8_t prod_16x8 = vmull_s8(s1_8x8, s2_8x8);
          dotprod = vpadalq_s16(dotprod, prod_16x8);
          col += (kWeightsPerNeonLane >> 1);
        }
        for (; col < m_cols; ++col)
        {
          postable_sum += row_ptr[col] * aligned_vec[col];
        } // for col
      }
      // Add the 4 intermediate sum values to get the final dot-prod value for
      // this row.
      int64x2_t pairwiseAdded = vpaddlq_s32(dotprod);
      int32_t neon_sum = vgetq_lane_s64(pairwiseAdded, 0) + vgetq_lane_s64(pairwiseAdded, 1);

      *result += ((neon_sum + postable_sum) * batch_scaling_factor);
    } // for row
  }   // for batch

  if (unaligned)
  {
    free(aligned_row_free);
  }
  free(aligned_vec_free);
}

void NeonMatrixBatchVectorMultiplyAccumulate(const float *matrix, int m_rows, int m_cols,
                                             const float *vector, int n_batch, float *result,
                                             int result_stride)
{
  // If v_size is not divisible by kWeightsPerNeonLane, we cannot use the main
  // vectorized loop, and we need to process sequentially. postamble_start shows
  // the start index where this should happen.
  const int postamble_start = m_cols - (m_cols & (kFloatWeightsPerNeonLane - 1));

  for (int b = 0; b < n_batch; b++)
  {
    float *result_in_batch = result + b * m_rows * result_stride;
    const float *vector_in_batch = vector + b * m_cols;
    const float *matrix_row = matrix;

    // Main matrix by vector multiplication loop
    for (int r = 0; r < m_rows; r++)
    {
      float32x4_t acc_32x4 = vmovq_n_f32(0.0);
      for (int c = 0; c < postamble_start; c += kFloatWeightsPerNeonLane)
      {
        // Load 4 float values from vector and matrix row.
        float32x4_t vector_f32x4 = vld1q_f32(vector_in_batch + c);
        float32x4_t matrix_f32x4 = vld1q_f32(matrix_row + c);
        // Multiply the vector and matrix row and add to accumulator.
        acc_32x4 = vmlaq_f32(acc_32x4, matrix_f32x4, vector_f32x4);
      }
      // Add the 4 intermediate sum values to get the final dot-prod value for
      // this column.
      *result_in_batch += (vgetq_lane_f32(acc_32x4, 0) + vgetq_lane_f32(acc_32x4, 1) +
                           vgetq_lane_f32(acc_32x4, 2) + vgetq_lane_f32(acc_32x4, 3));
      for (int c = postamble_start; c < m_cols; c++)
      {
        *result_in_batch += matrix_row[c] * vector_in_batch[c];
      }
      matrix_row += m_cols;
      result_in_batch += result_stride;
    }
  }
}

void NeonMatrixBatchVectorMultiplyAccumulate(const int8_t *__restrict__ matrix, const int m_rows,
                                             const int m_cols, const int8_t *__restrict__ vectors,
                                             const float *scaling_factors, int n_batch,
                                             int32_t *scratch, float *__restrict__ result,
                                             int result_stride)
{
  if (m_rows % 4 == 0 && result_stride == 1)
  {
    const int32_t *bias = static_cast<const int32_t *>(nullptr);
    NeonCpuBackendGemm(vectors, bias, matrix, n_batch, m_cols, m_rows,
                       /*output_zp =*/0, scratch);

    // Multiply by float scaling factors and write to result
    const int total_size = n_batch * m_rows;
    int i = 0;
    for (; i <= total_size - 8; i += 8, result += 8 * result_stride)
    {
      const float batch_scaling_factor0 = scaling_factors[i / m_rows];
      const float batch_scaling_factor1 = scaling_factors[(i + 4) / m_rows];
      const float32x4_t scaling_factor0 = vdupq_n_f32(batch_scaling_factor0);
      const float32x4_t scaling_factor1 = vdupq_n_f32(batch_scaling_factor1);
      const int32x4_t scratch_val0 = vld1q_s32(scratch + i);
      const int32x4_t scratch_val1 = vld1q_s32(scratch + i + 4);
      const float32x4_t float_val0 = vcvtq_f32_s32(scratch_val0);
      const float32x4_t float_val1 = vcvtq_f32_s32(scratch_val1);
      const float32x4_t result0 = vmlaq_f32(vld1q_f32(result), float_val0, scaling_factor0);
      const float32x4_t result1 =
          vmlaq_f32(vld1q_f32(result + 4 * result_stride), float_val1, scaling_factor1);
      vst1q_f32(result, result0);
      vst1q_f32(result + 4 * result_stride, result1);
    }
    scratch += i;
    for (; i < total_size; i++, result += result_stride)
    {
      const float batch_scaling_factor = scaling_factors[i / m_rows];
      int32_t x = *(scratch++);
      *result += x * batch_scaling_factor;
    }
    return;
  }
  NeonMatrixBatchVectorMultiplyAccumulate(matrix, m_rows, m_cols, vectors, scaling_factors, n_batch,
                                          result, result_stride);
}

} // namespace cker
} // namespace nnfw

#endif // USE_NEON

#endif // __NNFW_CKER_NEON_TENSOR_UTILS_H__
