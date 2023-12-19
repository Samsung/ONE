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
#include "cker/Types.h"
#include "cker/neon/neon_check.h"
#include "cker/ruy/RuySupport.h"
#include "util/logging.h"
#if defined __linux__ && defined __aarch64__
#include <sys/auxv.h>
#endif

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

constexpr int kFloatValuesPerNeonVector = 4;

// TODO(ahentz): Clean up.
using int8 = std::int8_t;
using uint8 = std::uint8_t;
using int16 = std::int16_t;
using uint16 = std::uint16_t;
using int32 = std::int32_t;
using uint32 = std::uint32_t;

template <int PerNeonSize> inline int RoundDownVectors(int size)
{
  return size & ~(PerNeonSize - 1);
}

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

inline int32_t AccumulateNeonLane(const int32x4_t lane)
{
#ifdef __aarch64__
  return vaddvq_s32(lane);
#else
  int64x2_t pairwiseAdded = vpaddlq_s32(lane);
  return vgetq_lane_s64(pairwiseAdded, 0) + vgetq_lane_s64(pairwiseAdded, 1);
#endif
}

} // namespace

// The implementation of dotprod detection is copied from ruy's internal
// function DetectDotprod().
// At the moment it's only implemented on Linux ARM64. Consider syncing again
// with ruy in the future to share improvements.
#if defined __linux__ && defined __aarch64__
inline bool DetectDotprodByLinuxAuxvMethod()
{
  // This is the value of HWCAP_ASIMDDP in sufficiently recent Linux headers,
  // however we need to support building against older headers for the time
  // being.
  const int kLocalHwcapAsimddp = 1 << 20;
  return getauxval(AT_HWCAP) & kLocalHwcapAsimddp;
}
#endif

inline bool DetectArmNeonDotprod()
{
#if defined __linux__ && defined __aarch64__
  return DetectDotprodByLinuxAuxvMethod();
#endif

  return false;
}

inline bool HasSdotInstruction()
{
  static const bool has_dotprod = DetectArmNeonDotprod();
  return has_dotprod;
}

#ifdef __aarch64__
// We interleave vector data to make the dot product logic more efficient.
// Suppose that vectors is:
//     a0 a1 a2 a3 a4 a5 ...
//     b0 b1 b2 b3 b4 b5 ...
//     c0 c1 c2 c3 c4 c5 ...
//     d0 d1 d2 d3 d4 d5 ...
//     e0 e1 e2 e3 e4 e5 ...
// This code interleaves them like this:
//     a0 a1 a2 a3 b0 b1 b2 b3 c0 c1 c2 c3 d0 d1 d2 d3 a4 a5 a6 a7 b4 ...
//     e0 e1 e2 e3 f0 f1 f2 f3 ...
// Once the data is interleaved, each 16-byte read from the vectors pointer
// contains 4 bytes from each of 4 vectors.
inline const int8_t *ShuffleVectors(const int8_t *vectors, const int n_batch, const int m_cols,
                                    void **shuffled_vectors_free)
{
  const int kWeightsPerUint32 = 4;

  int8 *shuffled_vectors = reinterpret_cast<int8 *>(
    aligned_alloc(kWeightsPerUint32, n_batch * m_cols, shuffled_vectors_free));

  for (int i = 0; i < n_batch; i += 4)
  {
    int8 *shuffled_vectors_ptr = shuffled_vectors + (i * m_cols);
    const int8 *unshuffled_vec0_ptr = reinterpret_cast<const int8 *>(vectors) + (i * m_cols);
    const int8 *unshuffled_vec1_ptr = reinterpret_cast<const int8 *>(vectors) + ((i + 1) * m_cols);
    const int8 *unshuffled_vec2_ptr = reinterpret_cast<const int8 *>(vectors) + ((i + 2) * m_cols);
    const int8 *unshuffled_vec3_ptr = reinterpret_cast<const int8 *>(vectors) + ((i + 3) * m_cols);
    const int8 *const end_vec0_ptr = unshuffled_vec1_ptr;

    while (unshuffled_vec0_ptr != end_vec0_ptr)
    {
      asm volatile(
        // This code path requires that (n_cols % 16) == 0 so we can safely
        // read in 16-byte chunks from each row.
        "ld1 {v0.16b}, [%[unshuffled_vec0_ptr]], #16\n"
        "ld1 {v1.16b}, [%[unshuffled_vec1_ptr]], #16\n"
        "ld1 {v2.16b}, [%[unshuffled_vec2_ptr]], #16\n"
        "ld1 {v3.16b}, [%[unshuffled_vec3_ptr]], #16\n"

        "st4 {v0.s, v1.s, v2.s, v3.s}[0], [%[shuffled_vectors_ptr]], #16\n"
        "st4 {v0.s, v1.s, v2.s, v3.s}[1], [%[shuffled_vectors_ptr]], #16\n"
        "st4 {v0.s, v1.s, v2.s, v3.s}[2], [%[shuffled_vectors_ptr]], #16\n"
        "st4 {v0.s, v1.s, v2.s, v3.s}[3], [%[shuffled_vectors_ptr]], #16\n"

        : [unshuffled_vec0_ptr] "+r"(unshuffled_vec0_ptr),
          [unshuffled_vec1_ptr] "+r"(unshuffled_vec1_ptr),
          [unshuffled_vec2_ptr] "+r"(unshuffled_vec2_ptr),
          [unshuffled_vec3_ptr] "+r"(unshuffled_vec3_ptr),
          [shuffled_vectors_ptr] "+r"(shuffled_vectors_ptr)
        :
        : "v0", "v1", "v2", "v3", "cc", "memory");
    }
  }

  return reinterpret_cast<const int8_t *>(shuffled_vectors);
}

// Notes about the speed of this version vs. the baseline (from memory):
// - With 256K of L1, we can keep a lot of vectors in cache.
//   I recall a reasonable speedup just by rearranging the loop to have
//   row on the outside and batch on the inside.
// - I also recall getting a nice speedup from sdot.
// - I tried many times to do better than the current implementation, using
//   loop unrolling and instruction reordering to avoid stalls, etc.
//   but I was not able to do significantly better. This code is, however,
//   much worse than what the processor spec sheet suggests is possible.
static void DotprodMatrixBatchFourVectorMultiplyAccumulate(const int8_t *__restrict__ matrix,
                                                           const int m_rows, const int m_cols,
                                                           const int8_t *vectors,
                                                           const float *scaling_factors,
                                                           int n_batch, float *__restrict__ result)
{
  void *shuffled_vectors_free;

  const int8_t *shuffled_vectors = ShuffleVectors(vectors, n_batch, m_cols, &shuffled_vectors_free);

  for (int row = 0; row < m_rows; row += 2)
  {
    for (int batch = 0; batch < n_batch; batch += 4)
    {
      float *result_ptr = result + (batch * m_rows) + row;
      const int8 *mat_ptr0 = matrix + (row * m_cols);
      const int8 *mat_ptr1 = matrix + ((row + 1) * m_cols);
      const int8 *mat_ptr0_end = mat_ptr1;
      const int8 *vec_ptr = shuffled_vectors + (batch * m_cols);
      const float *scaling_factors_ptr = scaling_factors + batch;
      const uint64_t wide_rows = m_rows * sizeof(float);
      const int8 *mat_ptr2 = matrix + ((row + 2) * m_cols);
      const int8 *mat_ptr3 = matrix + ((row + 3) * m_cols);

      asm volatile(
        // Zero out the accumulator registers.
        "dup v0.4s, wzr\n"
        "dup v1.4s, wzr\n"
        "dup v2.4s, wzr\n"
        "dup v3.4s, wzr\n"

        "1:\n" // batch_cols_loop

        // Read 16 more bytes from a pair of matrix rows.
        "ld1 {v12.16b}, [%[mat_ptr0]], #16\n"

        // Prefetch two rows ahead.
        "prfm pldl1strm, [%[mat_ptr2]]\n"
        "prfm pldl1strm, [%[mat_ptr3]]\n"

        // Read from input vectors 4 times; 64 bytes total.
        // Each 16-byte register contains parts of 4 vectors; see the
        // shuffle logic above.

        // From Benoit, places to look in the future:
        // - Move load instructions further from sdot
        // - Switch loop use-then-reload
        // - Do partial unrolling to use register space better
        "ld1 {v8.16b}, [%[vec_ptr]], #16\n"
        ".word 0x4f8ce100  // sdot v0.4s, v8.16b, v12.4b[0]\n"
        "ld1 {v9.16b}, [%[vec_ptr]], #16\n"
        ".word 0x4face121  // sdot v1.4s, v9.16b, v12.4b[1]\n"
        "ld1 {v10.16b}, [%[vec_ptr]], #16\n"
        ".word 0x4f8ce940  // sdot v0.4s, v10.16b, v12.4b[2]\n"
        "ld1 {v11.16b}, [%[vec_ptr]], #16\n"
        ".word 0x4face961  // sdot v1.4s, v11.16b, v12.4b[3]\n"

        // Update prefetch pointers.
        "add %[mat_ptr2], %[mat_ptr2], #16\n"
        "add %[mat_ptr3], %[mat_ptr3], #16\n"

        // Re-use those vectors for the next row as well.
        "ld1 {v13.16b}, [%[mat_ptr1]], #16\n"
        ".word 0x4f8de102  // sdot v2.4s, v8.16b, v13.4b[0]\n"
        ".word 0x4fade123  // sdot v3.4s, v9.16b, v13.4b[1]\n"
        ".word 0x4f8de942  // sdot v2.4s, v10.16b, v13.4b[2]\n"
        ".word 0x4fade963  // sdot v3.4s, v11.16b, v13.4b[3]\n"

        // If we're not done with these rows, continue.
        "cmp %[mat_ptr0], %[mat_ptr0_end]\n"
        "bne 1b\n" // batch_cols_loop

        // Done with the rows, sum the results.
        "add v0.4s, v0.4s, v1.4s\n"
        "add v2.4s, v2.4s, v3.4s\n"

        // Convert the per-vector sums to floating point.
        "scvtf v0.4s, v0.4s\n"
        "scvtf v1.4s, v2.4s\n"

        // Fetch scale factors.
        "ld1 {v4.4s}, [%[scaling_factors_ptr]]\n"

        // Multiply scale factors times sums.
        "fmul v0.4s, v4.4s, v0.4s\n"
        "fmul v1.4s, v4.4s, v1.4s\n"

        // Load previous result values.
        // The result position is:
        //   result[batch * m_rows + row]
        // Here that is factored into:
        //   result_ptr = result + row
        //   *result_ptr = res[0]
        //   (uint8*)result_ptr += (m_rows * sizeof(float))
        //   *result_ptr = res[1]
        //   ...
        // Since we're reading two rows at a time, though, we read both
        //   result[batch * m_rows + row]
        // and
        //   result[batch * m_rows + row + 1]
        "ld2 {v9.s, v10.s}[0], [%[result_ptr]], %[wide_rows]\n"
        "ld2 {v9.s, v10.s}[1], [%[result_ptr]], %[wide_rows]\n"
        "ld2 {v9.s, v10.s}[2], [%[result_ptr]], %[wide_rows]\n"
        "ld2 {v9.s, v10.s}[3], [%[result_ptr]], %[wide_rows]\n"

        // Go back to the starting position (subtract wide_rows * 4).
        "sub %[result_ptr], %[result_ptr], %[wide_rows], lsl #2\n"

        // Add previous result values.
        "fadd v9.4s, v9.4s, v0.4s\n"
        "fadd v10.4s, v10.4s, v1.4s\n"

        // Store results.
        "st2 {v9.s, v10.s}[0], [%[result_ptr]], %[wide_rows]\n"
        "st2 {v9.s, v10.s}[1], [%[result_ptr]], %[wide_rows]\n"
        "st2 {v9.s, v10.s}[2], [%[result_ptr]], %[wide_rows]\n"
        "st2 {v9.s, v10.s}[3], [%[result_ptr]], %[wide_rows]\n"
        : [mat_ptr0] "+r"(mat_ptr0), [mat_ptr1] "+r"(mat_ptr1), [vec_ptr] "+r"(vec_ptr),
          [result_ptr] "+r"(result_ptr), [mat_ptr2] "+r"(mat_ptr2), [mat_ptr3] "+r"(mat_ptr3)
        : [mat_ptr0_end] "r"(mat_ptr0_end), [scaling_factors_ptr] "r"(scaling_factors_ptr),
          [wide_rows] "r"(wide_rows)
        : "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
          "v13", "cc", "memory");
    }
  }

  free(shuffled_vectors_free);
}

static void DotprodMatrixBatchFourVectorMultiplyAccumulate(
  const int8_t *__restrict__ matrix, const int m_rows, const int m_cols, const int8_t *vectors,
  const float *scaling_factors, int n_batch, float *__restrict__ result,
  const float *per_channel_scale, const int32_t *input_offset, int32_t *row_sums)
{
  void *shuffled_vectors_free;
  const int8_t *shuffled_vectors = ShuffleVectors(vectors, n_batch, m_cols, &shuffled_vectors_free);

  for (int row = 0; row < m_rows; row += 2)
  {
    const float *channel_scales_ptr = per_channel_scale + row;
    int32_t *row_sums_ptr = row_sums ? row_sums + row : nullptr;
    for (int batch = 0; batch < n_batch; batch += 4)
    {
      float *result_ptr = result + (batch * m_rows) + row;
      const int8 *mat_ptr0 = matrix + (row * m_cols);
      const int8 *mat_ptr1 = matrix + ((row + 1) * m_cols);
      const int8 *mat_ptr0_end = mat_ptr1;
      const int8 *vec_ptr = shuffled_vectors + (batch * m_cols);
      const float *scaling_factors_ptr = scaling_factors + batch;
      const uint64_t wide_rows = m_rows * sizeof(float);
      const int32_t *batch_offsets_ptr = input_offset + batch;
      const int32_t is_channel_scale_nullptr = per_channel_scale == nullptr;
      const int32_t is_row_sums_nullptr = row_sums_ptr == nullptr;
      asm volatile(
        "dup v0.4s, wzr\n"
        "dup v1.4s, wzr\n"
        "dup v2.4s, wzr\n"
        "dup v3.4s, wzr\n"
        // Load zero points.
        "ld1 {v7.4s}, [%[batch_offsets_ptr]]\n"
        "ld1 {v4.4s}, [%[scaling_factors_ptr]]\n"
        // Zero out zero point accumulators.
        "dup v14.4s, wzr\n"
        "dup v15.4s, wzr\n"

        // Load per channel scales if not null.
        "cmp %w[is_channel_scale_nullptr], #0\n"
        "bne 1f\n"
        "ld1r {v16.4s}, [%[channel_scales_ptr]], #4\n"
        "ld1r {v17.4s}, [%[channel_scales_ptr]]\n"
        "fmul v16.4s, v16.4s, v4.4s\n"
        "fmul v17.4s, v17.4s, v4.4s\n"
        "b 2f\n"
        "1:\n"
        "mov v16.16b, v4.16b\n"
        "mov v17.16b, v4.16b\n"
        "2:\n"
        "ld1 {v12.16b}, [%[mat_ptr0]], #16\n"
        "ld1 {v8.16b}, [%[vec_ptr]], #16\n"
        ".word 0x4f8ce100  // sdot v0.4s, v8.16b, v12.4b[0]\n"
        "ld1 {v9.16b}, [%[vec_ptr]], #16\n"
        ".word 0x4face121  // sdot v1.4s, v9.16b, v12.4b[1]\n"
        "ld1 {v10.16b}, [%[vec_ptr]], #16\n"
        ".word 0x4f8ce940  // sdot v0.4s, v10.16b, v12.4b[2]\n"
        "ld1 {v11.16b}, [%[vec_ptr]], #16\n"
        ".word 0x4face961  // sdot v1.4s, v11.16b, v12.4b[3]\n"
        "ld1 {v13.16b}, [%[mat_ptr1]], #16\n"
        ".word 0x4f8de102  // sdot v2.4s, v8.16b, v13.4b[0]\n"
        ".word 0x4fade123  // sdot v3.4s, v9.16b, v13.4b[1]\n"
        ".word 0x4f8de942  // sdot v2.4s, v10.16b, v13.4b[2]\n"
        ".word 0x4fade963  // sdot v3.4s, v11.16b, v13.4b[3]\n"
        "cmp %w[is_row_sums_nullptr], #1\n"
        "bne 3f\n"
        // Accumulate row_sums for zero point calculations.
        "saddlp v12.8h, v12.16b\n"
        "saddlp v13.8h, v13.16b\n"
        "sadalp v14.4s, v12.8h\n"
        "sadalp v15.4s, v13.8h\n"
        "3:\n"
        "cmp %[mat_ptr0], %[mat_ptr0_end]\n"
        "bne 2b\n"
        "add v0.4s, v0.4s, v1.4s\n"
        "add v2.4s, v2.4s, v3.4s\n"

        "cmp %w[is_row_sums_nullptr], #1\n"
        "bne 4f\n"
        // Calculate zero point offsets.
        "addv s14, v14.4s\n"
        "addv s15, v15.4s\n"
        "dup v14.4s, v14.s[0]\n"
        "dup v15.4s, v15.s[0]\n"
        "b 5f\n"
        "4:\n"
        "ld1r {v14.4s}, [%[row_sums_ptr]], #4\n"
        "ld1r {v15.4s}, [%[row_sums_ptr]]\n"
        "5:\n"

        "mul v14.4s, v14.4s, v7.4s\n"
        "mul v15.4s, v15.4s, v7.4s\n"
        "sub v0.4s, v0.4s, v14.4s\n"
        "sub v2.4s, v2.4s, v15.4s\n"

        "scvtf v0.4s, v0.4s\n"
        "scvtf v1.4s, v2.4s\n"

        // Multiply scale.
        "fmul v0.4s, v16.4s, v0.4s\n"
        "fmul v1.4s, v17.4s, v1.4s\n"

        "ld2 {v9.s, v10.s}[0], [%[result_ptr]], %[wide_rows]\n"
        "ld2 {v9.s, v10.s}[1], [%[result_ptr]], %[wide_rows]\n"
        "ld2 {v9.s, v10.s}[2], [%[result_ptr]], %[wide_rows]\n"
        "ld2 {v9.s, v10.s}[3], [%[result_ptr]], %[wide_rows]\n"
        "sub %[result_ptr], %[result_ptr], %[wide_rows], lsl #2\n"
        "fadd v9.4s, v9.4s, v0.4s\n"
        "fadd v10.4s, v10.4s, v1.4s\n"
        "st2 {v9.s, v10.s}[0], [%[result_ptr]], %[wide_rows]\n"
        "st2 {v9.s, v10.s}[1], [%[result_ptr]], %[wide_rows]\n"
        "st2 {v9.s, v10.s}[2], [%[result_ptr]], %[wide_rows]\n"
        "st2 {v9.s, v10.s}[3], [%[result_ptr]], %[wide_rows]\n"
        : [mat_ptr0] "+r"(mat_ptr0), [mat_ptr1] "+r"(mat_ptr1), [vec_ptr] "+r"(vec_ptr),
          [result_ptr] "+r"(result_ptr), [row_sums_ptr] "+r"(row_sums_ptr)
        : [mat_ptr0_end] "r"(mat_ptr0_end), [scaling_factors_ptr] "r"(scaling_factors_ptr),
          [wide_rows] "r"(wide_rows), [channel_scales_ptr] "r"(channel_scales_ptr),
          [batch_offsets_ptr] "r"(batch_offsets_ptr),
          [is_channel_scale_nullptr] "r"(is_channel_scale_nullptr),
          [is_row_sums_nullptr] "r"(is_row_sums_nullptr)
        : "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
          "v13", "v14", "v15", "v16", "v17", "w0", "w1", "cc", "memory");
    }
  }

  free(shuffled_vectors_free);
}

// The DotprodMatrixBatchFourVectorMultiplyAccumulate kernel processes 4
// vectors in the same time as the baseline processes 1 vector. However, it
// requires 4 vectors of input.
//
// To take advantage of this speed difference, we add some zero-valued
// vectors to the batch so that n_batch is a multiple of 4. Then we execute
// DotprodMatrixBatchPaddedFourVectorMultiplyAccumulate on that padded batch,
// then extract just the results we want at the end (ignoring the extra padding
// outputs).
//
// The relative cost of the padding is large when the matrix is smaller than
// 128x128, so we don't use this code path on small matrices. On larger
// matrices, the computation cost dwarfs the padding cost, making this code
// viable.
//
// If we ignore the cost of padding, this kernel is:
//    1x the speed of NeonMatrixBatchVectorMultiplyImpl for n_batch = 1
//    2x the speed of NeonMatrixBatchVectorMultiplyImpl for n_batch = 2
//    3x the speed of NeonMatrixBatchVectorMultiplyImpl for n_batch = 3
//    ...
//
// We don't use this kernel when n_batch = 1 because the baseline kernel
// is fine for that case.
inline void DotprodMatrixBatchPaddedFourVectorMultiplyAccumulate(
  const int8_t *__restrict__ matrix, const int m_rows, const int m_cols, const int8_t *vectors,
  const float *scaling_factors, int n_batch, float *__restrict__ result,
  const float *per_channel_scale, const int32_t *input_offset, int32_t *row_sums)
{
  const int kWeightsPerUint32 = 4;

  // Round to the nearest multiple of 4.
  int batch_round_up = n_batch;
  if (n_batch % 4 != 0)
  {
    batch_round_up += (4 - n_batch % 4);
  }
  assert(n_batch <= batch_round_up);

  void *padded_vectors_free;
  const int padded_vectors_size = batch_round_up * m_cols;
  int8_t *padded_vectors = reinterpret_cast<int8_t *>(
    aligned_alloc(kWeightsPerUint32, padded_vectors_size, &padded_vectors_free));
  memset(padded_vectors, 0, padded_vectors_size);

  void *padded_result_free;
  const int result_size = n_batch * m_rows * sizeof(float);
  const int padded_result_size = batch_round_up * m_rows * sizeof(float);
  float *padded_result = reinterpret_cast<float *>(
    aligned_alloc(kWeightsPerUint32, padded_result_size, &padded_result_free));
  memcpy(padded_result, result, result_size);
  memset(reinterpret_cast<char *>(padded_result) + result_size, 0,
         padded_result_size - result_size);

  // Copy the input into the padded data structure.
  assert(n_batch * m_cols <= padded_vectors_size);
  memcpy(padded_vectors, vectors, n_batch * m_cols);

  void *padded_scaling_factors_free;
  const int padded_scaling_factors_size = batch_round_up * sizeof(float);
  float *padded_scaling_factors = reinterpret_cast<float *>(
    aligned_alloc(kWeightsPerUint32, padded_scaling_factors_size, &padded_scaling_factors_free));
  assert(static_cast<int>(n_batch * sizeof(float)) <= padded_scaling_factors_size);
  assert(static_cast<int>(batch_round_up * sizeof(float)) <= padded_scaling_factors_size);
  memset(padded_scaling_factors, 0, batch_round_up * sizeof(float));
  memcpy(padded_scaling_factors, scaling_factors, n_batch * sizeof(float));

  if (input_offset != nullptr)
  {
    void *padded_input_offset_free;
    const int padded_input_offset_size = batch_round_up * sizeof(int32_t);
    int32_t *padded_input_offset = reinterpret_cast<int32_t *>(
      aligned_alloc(kWeightsPerUint32, padded_input_offset_size, &padded_input_offset_free));
    assert(static_cast<int>(n_batch * sizeof(int32_t)) <= padded_input_offset_size);
    assert(static_cast<int>(batch_round_up * sizeof(int32_t)) <= padded_input_offset_size);
    memset(padded_input_offset, 0, batch_round_up * sizeof(int32_t));
    memcpy(padded_input_offset, input_offset, n_batch * sizeof(int32_t));

    // Call the main kernel.
    DotprodMatrixBatchFourVectorMultiplyAccumulate(
      matrix, m_rows, m_cols, padded_vectors, padded_scaling_factors, batch_round_up, padded_result,
      per_channel_scale, padded_input_offset, row_sums);

    free(padded_input_offset_free);
  }
  else
  {
    // Call the main kernel.
    DotprodMatrixBatchFourVectorMultiplyAccumulate(matrix, m_rows, m_cols, padded_vectors,
                                                   padded_scaling_factors, batch_round_up,
                                                   padded_result);
  }
  memcpy(result, padded_result, result_size);

  free(padded_result_free);
  free(padded_vectors_free);
  free(padded_scaling_factors_free);
}

inline void DotprodMatrixBatchPaddedFourVectorMultiplyAccumulate(
  const int8_t *__restrict__ matrix, const int m_rows, const int m_cols, const int8_t *vectors,
  const float *scaling_factors, int n_batch, float *__restrict__ result)
{
  DotprodMatrixBatchPaddedFourVectorMultiplyAccumulate(
    matrix, m_rows, m_cols, vectors, scaling_factors, n_batch, result,
    /*per_channel_scale=*/nullptr, /*input_offset=*/nullptr,
    /*row_sums=*/nullptr);
}
#endif // __aarch64__

inline void NeonCwiseClipping(float *vector, const int v_size, const float clipping_value)
{
  const float32x4_t clipping_value_f32x4 = vmovq_n_f32(clipping_value);
  const float32x4_t neg_clipping_value_f32x4 = vmovq_n_f32(-clipping_value);

  int i = 0;
  for (; i <= v_size - kFloatValuesPerNeonVector; i += kFloatValuesPerNeonVector)
  {
    // Load from memory to vector.
    float32x4_t v_f32x4 = vld1q_f32(vector + i);
    // Clip between clipping_value and -clipping_value.
    v_f32x4 = vminq_f32(clipping_value_f32x4, v_f32x4);
    v_f32x4 = vmaxq_f32(neg_clipping_value_f32x4, v_f32x4);
    // Save to output.
    vst1q_f32(vector + i, v_f32x4);
  }
  for (; i < v_size; i++)
  {
    vector[i] = std::max(std::min(clipping_value, vector[i]), -clipping_value);
  }
}

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

inline void NeonCpuBackendGemm(const int8_t *input, const int32_t *bias,
                               const int8_t *input_to_gate_weights, int32_t n_batch,
                               int32_t n_input, int32_t n_output, int32_t, int32_t *scratch,
                               ruy::Context *ruy_context)
{
  MatrixParams<int8_t> lhs_params;
  lhs_params.order = Order::kRowMajor;
  lhs_params.rows = n_output;
  lhs_params.cols = n_input;
  lhs_params.cache_policy = CachePolicy::kAlwaysCache;

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
  ruy::Matrix<int8_t> ruy_lhs;
  ruy::Matrix<int8_t> ruy_rhs;
  ruy::Matrix<int32_t> ruy_dst;
  // Note that cache is always enabled for input and weight tensors
  ruy_support::MakeRuyMatrix(lhs_params, input_to_gate_weights, &ruy_lhs, true);
  ruy_support::MakeRuyMatrix(rhs_params, input, &ruy_rhs, true);
  ruy_support::MakeRuyMatrix(dst_params, scratch, &ruy_dst);

  ruy::MulParams<int32_t, int32_t> ruy_mul_params;
  ruy_support::MakeRuyMulParams(gemm_params, &ruy_mul_params);

  ruy::Mul(ruy_lhs, ruy_rhs, ruy_mul_params, ruy_context, &ruy_dst);
}

inline void NeonSub1Vector(const float *vector, int v_size, float *result)
{
  // If v_size is not divisible by the vector size, then we need to process the
  // final few elements sequentially. postamble_start shows the start index
  // where this should happen.
  const int postamble_start = RoundDownVectors<kFloatValuesPerNeonVector>(v_size);

  float32x4_t one_f32x4 = vmovq_n_f32(1.0);
  int v = 0;
  for (; v < postamble_start; v += kFloatValuesPerNeonVector)
  {
    // Load 4 float values from the current pointers of the input column and
    // subtract from 1.
    float32x4_t v_f32x4 = vld1q_f32(vector + v);
    float32x4_t result_f32x4 = vsubq_f32(one_f32x4, v_f32x4);
    // Save to output.
    vst1q_f32(result + v, result_f32x4);
  }
  for (; v < v_size; v++)
  {
    result[v] = 1.0f - vector[v];
  }
}

inline void NeonSymmetricQuantizeFloats(const float *values, const int size,
                                        int8_t *quantized_values, float *min, float *max,
                                        float *scaling_factor)
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

inline void NeonMatrixBatchVectorMultiplyAccumulate(const int8_t *__restrict__ matrix,
                                                    const int m_rows, const int m_cols,
                                                    const int8_t *__restrict__ vectors,
                                                    const float *scaling_factors, int n_batch,
                                                    float *__restrict__ result, int result_stride)
{
#ifdef __aarch64__
  if (HasSdotInstruction() && m_cols % 16 == 0 && m_rows % 2 == 0 && m_rows >= n_batch)
  {
    if (n_batch % 4 == 0 && result_stride == 1)
    {
      // Benchmarks suggest that it's always better to use the batch code
      // when we can, even on small matrices.
      DotprodMatrixBatchFourVectorMultiplyAccumulate(matrix, m_rows, m_cols, vectors,
                                                     scaling_factors, n_batch, result);
      return;
    }
    else if (result_stride == 1 && n_batch >= 2 && m_rows * m_cols >= 128 * 128)
    {
      DotprodMatrixBatchPaddedFourVectorMultiplyAccumulate(matrix, m_rows, m_cols, vectors,
                                                           scaling_factors, n_batch, result);
      return;
    }
  }
#endif // __aarch64__

  static const int kWeightsPerUint32 = 4;
  static const int kWeightsPerNeonLane = 16;
  // Assuming *matrix is kWeightsPerUint32-byte aligned,
  // every row of the matrix is also
  // kWeightsPerUint32-byte aligned as long as cols is
  // a multiple of kWeightsPerUint32. The assumption
  // is currently satisfied by TFLite's 16-byte memory
  // alignment scheme.
  //
  // Otherwise, we allocate an aligned memory block and set
  // a flag to later copy rows from matrix to the block
  // for aligned multiplication.
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
  // vectorized loop, and we need to process sequentially. postamble_half_start
  // shows the start index where this should happen. Between postamble_start and
  // postamble_half_start we can still process kWeightsPerNeonLane >> 1 in a
  // vectorized form.
  const int postamble_half_start = m_cols & ~(kWeightsPerNeonLane - 1);
  const int postamble_start = m_cols & ~((kWeightsPerNeonLane >> 1) - 1);

  for (int batch = 0; batch < n_batch; ++batch)
  {
    const float batch_scaling_factor = scaling_factors[batch];
    // Copy the vector data to an aligned vector.
    memcpy(aligned_vec, vectors + batch * m_cols, sizeof(int8_t) * m_cols);
    // Compute dot-product for every column.
    for (int row = 0; row < m_rows; ++row, result += result_stride)
    {
      // Get the address of the first element of the row.
      int8_t *row_ptr = (int8_t *)matrix + row * m_cols; // NOLINT
      if (unaligned)
      {
        memcpy(aligned_row, row_ptr, sizeof(int8_t) * m_cols);
        row_ptr = aligned_row;
      }

      // Initialize the dot product sum for the row to 0.
      int32x4_t dotprod_32x4 = vmovq_n_s32(0);

      // Prefetch the row to cache.
      __builtin_prefetch(row_ptr, 0 /* prefetch for read */, 3 /* temporal locality */);

      // For every block of 16 8-bit elements.
      int col = 0;
      for (; col < postamble_half_start; col += kWeightsPerNeonLane)
      {
        // Load 16 8-bit values from the row and vector, each, to operate on.
        // Here the assumption is that each buffer is 4-byte aligned. Otherwise,
        // performance may suffer significantly.
        assert( // NOLINT
          ((uintptr_t)(&row_ptr[col]) & (kWeightsPerUint32 - 1)) == 0);
        const int8x16_t s1_8x16 = vld1q_s8((const int8_t *)(aligned_vec + col));
        const int8x16_t s2_8x16 = vld1q_s8((const int8_t *)(row_ptr + col));
        // Multiply the low bits (i.e. the lower 8 8bit numbers in the
        // registers).
        int16x8_t prod_16x8 = vmull_s8(vget_low_s8(s1_8x16), vget_low_s8(s2_8x16));
        // Multiply the high bits (i.e. the higher 8 8bit numbers in the
        // registers), and accumulate with the result of the low bits product.
        // The assumption here is that overflow will not happen as we quantize
        // our values to be in the range [-127, 127]. As such the sum of the 2
        // products is always strictly smaller than 15-bits (32767 in absolute
        // value).
        prod_16x8 = vmlal_s8(prod_16x8, vget_high_s8(s1_8x16), vget_high_s8(s2_8x16));

        dotprod_32x4 = vpadalq_s16(dotprod_32x4, prod_16x8);
      } // for col

      // Half iteration dealing only 8 elements
      // TODO(raziel): if (ABSL_PREDICT_FALSE(col < postamble_start))
      if (col < postamble_start)
      {
        // Load 8 8-bit values from the row and column each to operate on.
        // Here the assumption is that each buffer is 4-bytes aligned.
        // Otherwise, performance may suffer significantly.
        assert( // NOLINT
          ((uintptr_t)(&row_ptr[col]) & (kWeightsPerUint32 - 1)) == 0);
        const int8x8_t s1_8x8 = vld1_s8((const int8_t *)(aligned_vec + col));
        const int8x8_t s2_8x8 = vld1_s8((const int8_t *)(row_ptr + col));
        const int16x8_t prod_16x8 = vmull_s8(s1_8x8, s2_8x8);
        dotprod_32x4 = vpadalq_s16(dotprod_32x4, prod_16x8);
        col += (kWeightsPerNeonLane >> 1);
      }
      // Add the 4 intermediate sum values to get the final dot-prod value for
      // this row.
      int32_t dotprod = AccumulateNeonLane(dotprod_32x4);
      // Postamble loop.
      // TODO(raziel): if (ABSL_PREDICT_FALSE(col < m_cols))
      for (; col < m_cols; ++col)
      {
        dotprod += row_ptr[col] * aligned_vec[col];
      } // for col

      *result += dotprod * batch_scaling_factor;
    } // for row
  }   // for batch

  if (unaligned)
  {
    free(aligned_row_free);
  }
  free(aligned_vec_free);
}

inline void NeonMatrixBatchVectorMultiplyAccumulate(const float *matrix, int m_rows, int m_cols,
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

inline void NeonMatrixBatchVectorMultiplyAccumulate(const int8_t *__restrict__ matrix,
                                                    const int m_rows, const int m_cols,
                                                    const int8_t *__restrict__ vectors,
                                                    const float *scaling_factors, int n_batch,
                                                    int32_t *scratch, float *__restrict__ result,
                                                    int result_stride, ruy::Context *ruy_context)
{
  if (m_rows % 4 == 0 && result_stride == 1)
  {
    const int32_t *bias = static_cast<const int32_t *>(nullptr);
    NeonCpuBackendGemm(vectors, bias, matrix, n_batch, m_cols, m_rows,
                       /*output_zp =*/0, scratch, ruy_context);

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
