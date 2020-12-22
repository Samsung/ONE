/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __NNFW_CKER_UTILS_H__
#define __NNFW_CKER_UTILS_H__

#include "Shape.h"

#include "neon/neon_check.h"

#include <algorithm>
#include <cstdint>
#include <fixedpoint/fixedpoint.h>

namespace nnfw
{
namespace cker
{

template <typename T>
inline T ActivationFunctionWithMinMax(T x, T output_activation_min, T output_activation_max)
{
  return std::min<T>(std::max<T>(x, output_activation_min), output_activation_max);
}

inline void QuantizeMultiplier(double double_multiplier, int32_t *quantized_multiplier, int *shift)
{
  if (double_multiplier == 0.)
  {
    *quantized_multiplier = 0;
    *shift = 0;
    return;
  }

  const double q = std::frexp(double_multiplier, shift);
  auto q_fixed = static_cast<int64_t>(round(q * (1ll << 31)));

  assert(q_fixed <= (1ll << 31));
  if (q_fixed == (1ll << 31))
  {
    q_fixed /= 2;
    ++*shift;
  }
  assert(q_fixed <= std::numeric_limits<int32_t>::max());
  // A shift amount smaller than -31 would cause all bits to be shifted out
  // and thus all results would be zero. We implement that instead with
  // q_fixed==0, so as to avoid hitting issues with right-shift
  // operations with shift amounts greater than 31. Note that this happens
  // roughly when abs(double_multiplier) < 2^-31 and the present handling means
  // that we're effectively flushing tiny double_multiplier's to zero.
  // We could conceivably handle values in the range (roughly) [32, 63]
  // as 'denormals' i.e. (shift==0, q_fixed < 2^30). In that point of view
  // the present handling is just doing 'flush denormals to zero'. We could
  // reconsider and actually generate nonzero denormals if a need arises.
  if (*shift < -31)
  {
    *shift = 0;
    q_fixed = 0;
  }
  *quantized_multiplier = static_cast<int32_t>(q_fixed);
}

inline void QuantizeMultiplierSmallerThanOneExp(double double_multiplier,
                                                int32_t *quantized_multiplier, int *left_shift)
{
  assert(double_multiplier < 1.0);
  assert(double_multiplier > 0.0);
  int shift;
  QuantizeMultiplier(double_multiplier, quantized_multiplier, &shift);
  assert(shift <= 0);
  *left_shift = shift;
}

inline int32_t MultiplyByQuantizedMultiplier(int32_t x, int32_t quantized_multiplier, int shift)
{
  int left_shift = shift > 0 ? shift : 0;
  int right_shift = shift > 0 ? 0 : -shift;
  return gemmlowp::RoundingDivideByPOT(
    gemmlowp::SaturatingRoundingDoublingHighMul(x * (1 << left_shift), quantized_multiplier),
    right_shift);
}

inline int32_t MultiplyByQuantizedMultiplierGreaterThanOne(int32_t x, int32_t quantized_multiplier,
                                                           int left_shift)
{
  return gemmlowp::SaturatingRoundingDoublingHighMul(x * (1 << left_shift), quantized_multiplier);
}

inline int32_t MultiplyByQuantizedMultiplierSmallerThanOneExp(int32_t x,
                                                              int32_t quantized_multiplier,
                                                              int left_shift)
{
  return gemmlowp::RoundingDivideByPOT(
    gemmlowp::SaturatingRoundingDoublingHighMul(x, quantized_multiplier), -left_shift);
}

#ifdef USE_NEON
inline int32x4x4_t MultiplyByQuantizedMultiplier4Rows(int32x4x4_t input_val,
                                                      int32_t quantized_multiplier, int32_t shift)
{
  const int left_shift = std::max(shift, 0);
  const int right_shift = std::min(shift, 0);
  int32x4x4_t result;

  int32x4_t multiplier_dup = vdupq_n_s32(quantized_multiplier);
  int32x4_t left_shift_dup = vdupq_n_s32(left_shift);
  int32x4_t right_shift_dup = vdupq_n_s32(right_shift);

  result.val[0] = vrshlq_s32(
    vqrdmulhq_s32(vshlq_s32(input_val.val[0], left_shift_dup), multiplier_dup), right_shift_dup);

  result.val[1] = vrshlq_s32(
    vqrdmulhq_s32(vshlq_s32(input_val.val[1], left_shift_dup), multiplier_dup), right_shift_dup);

  result.val[2] = vrshlq_s32(
    vqrdmulhq_s32(vshlq_s32(input_val.val[2], left_shift_dup), multiplier_dup), right_shift_dup);

  result.val[3] = vrshlq_s32(
    vqrdmulhq_s32(vshlq_s32(input_val.val[3], left_shift_dup), multiplier_dup), right_shift_dup);

  return result;
}
#endif

inline int NodeOffset(int b, int h, int w, int height, int width)
{
  return (b * height + h) * width + w;
}

inline int CountLeadingZeros(uint32_t integer_input)
{
  const uint32_t one_in_leading_positive = 1U << 31;
  int leading_zeros = 0;
  while (integer_input < one_in_leading_positive)
  {
    integer_input <<= 1;
    ++leading_zeros;
  }
  return leading_zeros;
}

inline void GetInvSqrtQuantizedMultiplierExp(int32_t input, int reverse_shift,
                                             int32_t *output_inv_sqrt, int *output_shift)
{
  assert(input >= 0);
  if (input <= 1)
  {
    // Handle the input value 1 separately to avoid overflow in that case
    // in the general computation below (b/143972021). Also handle 0 as if it
    // were a 1. 0 is an invalid input here (divide by zero) and 1 is a valid
    // but rare/unrealistic input value. We can expect both to occur in some
    // incompletely trained models, but probably not in fully trained models.
    *output_inv_sqrt = std::numeric_limits<std::int32_t>::max();
    *output_shift = 0;
    return;
  }
  assert(input > 1);
  *output_shift = 11;
  while (input >= (1 << 29))
  {
    input /= 4;
    ++*output_shift;
  }
  const unsigned max_left_shift_bits = CountLeadingZeros(static_cast<uint32_t>(input)) - 1;
  const unsigned max_left_shift_bit_pairs = max_left_shift_bits / 2;
  const unsigned left_shift_bit_pairs = max_left_shift_bit_pairs - 1;
  *output_shift -= left_shift_bit_pairs;
  input <<= 2 * left_shift_bit_pairs;
  assert(input >= (1 << 27));
  assert(input < (1 << 29));
  using gemmlowp::FixedPoint;
  using gemmlowp::Rescale;
  using gemmlowp::SaturatingRoundingMultiplyByPOT;
  // Using 3 integer bits gives us enough room for the internal arithmetic in
  // this Newton-Raphson iteration.
  using F3 = FixedPoint<int32_t, 3>;
  using F0 = FixedPoint<int32_t, 0>;
  const F3 fixedpoint_input = F3::FromRaw(input >> 1);
  const F3 fixedpoint_half_input = SaturatingRoundingMultiplyByPOT<-1>(fixedpoint_input);
  const F3 fixedpoint_half_three =
    GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(F3, (1 << 28) + (1 << 27), 1.5);
  // Newton-Raphson iteration
  // Naive unoptimized starting guess: x = 1
  F3 x = F3::One();
  // Naive unoptimized number of iterations: 5
  for (int i = 0; i < 5; i++)
  {
    const F3 x3 = Rescale<3>(x * x * x);
    x = Rescale<3>(fixedpoint_half_three * x - fixedpoint_half_input * x3);
  }
  const F0 fixedpoint_half_sqrt_2 =
    GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(F0, 1518500250, std::sqrt(2.) / 2.);
  x = x * fixedpoint_half_sqrt_2;
  *output_inv_sqrt = x.raw();
  if (*output_shift < 0)
  {
    *output_inv_sqrt <<= -*output_shift;
    *output_shift = 0;
  }
  // Convert right shift (right is positive) to left shift.
  *output_shift *= reverse_shift;
}

// Comment from tensorflow lite:
//
// DO NOT USE THIS STRUCT FOR NEW FUNCTIONALITY BEYOND IMPLEMENTING
// BROADCASTING.
//
// NdArrayDesc<N> describes the shape and memory layout of an N-dimensional
// rectangular array of numbers.
//
// NdArrayDesc<N> is basically identical to Dims<N> defined in types.h.
// However, as Dims<N> is to be deprecated, this class exists as an adaptor
// to enable simple unoptimized implementations of element-wise broadcasting
// operations.
template <int N> struct NdArrayDesc
{
  // The "extent" of each dimension. Indices along dimension d must be in the
  // half-open interval [0, extents[d]).
  int extents[N];

  // The number of *elements* (not bytes) between consecutive indices of each
  // dimension.
  int strides[N];
};

// Comment from tensorflow lite:
//
// DO NOT USE THIS FUNCTION FOR NEW FUNCTIONALITY BEYOND IMPLEMENTING
// BROADCASTING.
//
// Same as Offset(), except takes as NdArrayDesc<N> instead of Dims<N>.
inline int SubscriptToIndex(const NdArrayDesc<4> &desc, int i0, int i1, int i2, int i3)
{
  assert(i0 >= 0 && i0 < desc.extents[0]);
  assert(i1 >= 0 && i1 < desc.extents[1]);
  assert(i2 >= 0 && i2 < desc.extents[2]);
  assert(i3 >= 0 && i3 < desc.extents[3]);
  return i0 * desc.strides[0] + i1 * desc.strides[1] + i2 * desc.strides[2] + i3 * desc.strides[3];
}

template <int N> inline int SubscriptToIndexGeneric(const NdArrayDesc<N> *desc, int *iter)
{
  int ret_indx = 0;
  for (size_t idx = 0; idx < static_cast<size_t>(N); idx++)
  {
    assert(iter[idx] >= 0 && iter[idx] < desc->extents[idx]);
    ret_indx += iter[idx] * desc->strides[idx];
  }

  return ret_indx;
}

// Copies dims to desc, calculating strides.
template <int N> inline void CopyDimsToDesc(const Shape &input_shape, NdArrayDesc<N> *desc_out)
{
  int desc_stride = 1;
  for (int i = N - 1; i >= 0; --i)
  {
    desc_out->extents[i] = input_shape.Dims(i);
    desc_out->strides[i] = desc_stride;
    desc_stride *= input_shape.Dims(i);
  }
}

template <int N>
inline void
NdArrayDescsForElementwiseBroadcast(const Shape &input0_shape, const Shape &input1_shape,
                                    NdArrayDesc<N> *desc0_out, NdArrayDesc<N> *desc1_out)
{
  assert(desc0_out != nullptr);
  assert(desc1_out != nullptr);

  auto extended_input0_shape = Shape::ExtendedShape(N, input0_shape);
  auto extended_input1_shape = Shape::ExtendedShape(N, input1_shape);

  // Copy dims to desc, calculating strides.
  CopyDimsToDesc<N>(extended_input0_shape, desc0_out);
  CopyDimsToDesc<N>(extended_input1_shape, desc1_out);

  // Walk over each dimension. If the extents are equal do nothing.
  // Otherwise, set the desc with extent 1 to have extent equal to the other and
  // stride 0.
  for (int i = 0; i < N; ++i)
  {
    const int extent0 = extended_input0_shape.Dims(i);
    const int extent1 = extended_input1_shape.Dims(i);
    if (extent0 != extent1)
    {
      if (extent0 == 1)
      {
        desc0_out->strides[i] = 0;
        desc0_out->extents[i] = extent1;
      }
      else
      {
        assert(extent1 == 1);
        desc1_out->strides[i] = 0;
        desc1_out->extents[i] = extent0;
      }
    }
  }
}

template <int N>
inline void
NdArrayDescsForElementwiseBroadcast(const Shape &input0_shape, const Shape &input1_shape,
                                    const Shape &input2_shape, NdArrayDesc<N> *desc0_out,
                                    NdArrayDesc<N> *desc1_out, NdArrayDesc<N> *desc2_out)
{
  assert(desc0_out != nullptr);
  assert(desc1_out != nullptr);
  assert(desc2_out != nullptr);

  auto extended_input0_shape = Shape::ExtendedShape(N, input0_shape);
  auto extended_input1_shape = Shape::ExtendedShape(N, input1_shape);
  auto extended_input2_shape = Shape::ExtendedShape(N, input2_shape);

  // Copy dims to desc, calculating strides.
  CopyDimsToDesc<N>(extended_input0_shape, desc0_out);
  CopyDimsToDesc<N>(extended_input1_shape, desc1_out);
  CopyDimsToDesc<N>(extended_input2_shape, desc2_out);

  // Walk over each dimension. If the extents are equal do nothing.
  // Otherwise, set the desc with extent 1 to have extent equal to the other and
  // stride 0.
  for (int i = 0; i < N; ++i)
  {
    const int extent0 = extended_input0_shape.Dims(i);
    const int extent1 = extended_input1_shape.Dims(i);
    const int extent2 = extended_input2_shape.Dims(i);

    int extent = extent0;
    if (extent1 != 1)
      extent = extent1;
    if (extent2 != 1)
      extent = extent2;

    assert(extent0 == 1 || extent0 == extent);
    assert(extent1 == 1 || extent1 == extent);
    assert(extent2 == 1 || extent2 == extent);

    if (!(extent0 == extent1 && extent1 == extent2))
    {
      if (extent0 == 1)
      {
        desc0_out->strides[i] = 0;
        desc0_out->extents[i] = extent;
      }
      if (extent1 == 1)
      {
        desc1_out->strides[i] = 0;
        desc1_out->extents[i] = extent;
      }
      if (extent2 == 1)
      {
        desc2_out->strides[i] = 0;
        desc2_out->extents[i] = extent;
      }
    }
  }
}

// Gets next index to iterate through a multidimensional array.
inline bool NextIndex(const int num_dims, const int *dims, int *current)
{
  if (num_dims == 0)
  {
    return false;
  }
  assert(dims != nullptr);
  assert(current != nullptr);
  int carry = 1;
  for (int idx = num_dims - 1; idx >= 0; --idx)
  {
    int current_val = current[idx] + carry;
    assert(dims[idx] >= current_val);
    if (dims[idx] == current_val)
    {
      current[idx] = 0;
    }
    else
    {
      current[idx] = current_val;
      carry = 0;
      break;
    }
  }
  return (carry == 0);
}

// Gets offset of index if reducing on axis. When reducing, the flattened offset
// will not change, if the input index changes on the given axis. For example,
// if you have a 3D tensor and you are reducing to 2D by eliminating axis 0,
// then index (0, 1, 2) and index (1, 1, 2) will map to the same flattened
// offset.
// TODO(kanlig): uses Dims to represent dimensions.
inline size_t ReducedOutputOffset(const int num_dims, const int *dims, const int *index,
                                  const int num_axis, const int *axis)
{
  if (num_dims == 0)
  {
    return 0;
  }

  assert(dims != nullptr);
  assert(index != nullptr);

  size_t offset = 0;
  for (int idx = 0; idx < num_dims; ++idx)
  {
    // if we need to skip this axis
    bool is_axis = false;
    if (axis != nullptr)
    {
      for (int axis_idx = 0; axis_idx < num_axis; ++axis_idx)
      {
        if (idx == axis[axis_idx])
        {
          is_axis = true;
          break;
        }
      }
    }
    if (!is_axis)
    {
      offset = offset * static_cast<size_t>(dims[idx]) + static_cast<size_t>(index[idx]);
    }
  }
  return offset;
}

template <typename T> void optimized_ops_preload_l1_keep(const T *ptr)
{
#ifdef __GNUC__
  // builtin offered by GCC-compatible compilers including clang
  __builtin_prefetch(ptr, /* 0 means read */ 0, /* 3 means high locality */ 3);
#else
  (void)ptr;
#endif
}

// Writes randomly accessed values from `input` sequentially into `output`.
template <typename T> class SequentialTensorWriter
{
public:
  SequentialTensorWriter(const T *input_data, T *output_data)
    : input_data_(input_data), output_ptr_(output_data)
  {
  }

  void Write(int position) { *output_ptr_++ = input_data_[position]; }
  void WriteN(int position, int len)
  {
    memcpy(output_ptr_, &input_data_[position], sizeof(T) * len);
    output_ptr_ += len;
  }

private:
  const T *input_data_;
  T *output_ptr_;
};

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_UTILS_H__
