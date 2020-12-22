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

#ifndef __NNFW_CKER_OPTIMIZED_BINARYARITHMETICOPS_H__
#define __NNFW_CKER_OPTIMIZED_BINARYARITHMETICOPS_H__

#include <functional>
#include <limits>
#include <utility>
#include "cker/neon/neon_check.h"
#include "cker/operation/reference/BinaryArithmeticOps.h"
#include "cker/Shape.h"
#include "cker/Types.h"
#include "cker/Utils.h"
#include "fixedpoint/fixedpoint.h"

namespace nnfw
{
namespace cker
{
namespace optimized
{

template <typename ElementwiseF, typename ScalarBroadcastF, typename T>
inline void BinaryBroadcastFiveFold(const BinaryArithmeticOpParam &params, bool switch_inputs,
                                    const Shape & /* unswitched_input1_shape */,
                                    const T *unswitched_input1_data,
                                    const Shape & /* unswitched_input2_shape */,
                                    const T *unswitched_input2_data,
                                    const Shape & /* output_shape */, T *output_data,
                                    ElementwiseF elementwise_f, ScalarBroadcastF scalar_broadcast_f)
{
  const T *input1_data = switch_inputs ? unswitched_input2_data : unswitched_input1_data;
  const T *input2_data = switch_inputs ? unswitched_input1_data : unswitched_input2_data;

  // Fivefold nested loops. The second input resets its position for each
  // iteration of the second loop. The first input resets its position at the
  // beginning of the fourth loop. The innermost loop is an elementwise add of
  // sections of the arrays.
  T *output_data_ptr = output_data;
  const T *input1_data_ptr = input1_data;
  const T *input2_data_reset = input2_data;
  // In the fivefold pattern, y0, y2 and y4 are not broadcast, and so shared
  // between input shapes. y3 for input 1 is always broadcast, and so the
  // dimension there is 1, whereas optionally y1 might be broadcast for input 2.
  // Put another way,
  // input1.shape.FlatSize = y0 * y1 * y2 * y4,
  // input2.shape.FlatSize = y0 * y2 * y3 * y4.
  int y0 = params.broadcast_shape[0];
  int y1 = params.broadcast_shape[1];
  int y2 = params.broadcast_shape[2];
  int y3 = params.broadcast_shape[3];
  int y4 = params.broadcast_shape[4];
  if (y4 > 1)
  {
    // General fivefold pattern, with y4 > 1 so there is a non-broadcast inner
    // dimension.
    for (int i0 = 0; i0 < y0; ++i0)
    {
      const T *input2_data_ptr = nullptr;
      for (int i1 = 0; i1 < y1; ++i1)
      {
        input2_data_ptr = input2_data_reset;
        for (int i2 = 0; i2 < y2; ++i2)
        {
          for (int i3 = 0; i3 < y3; ++i3)
          {
            elementwise_f(y4, params, input1_data_ptr, input2_data_ptr, output_data_ptr);
            input2_data_ptr += y4;
            output_data_ptr += y4;
          }
          // We have broadcast y4 of input1 data y3 times, and now move on.
          input1_data_ptr += y4;
        }
      }
      // We have broadcast y2*y3*y4 of input2 data y1 times, and now move on.
      input2_data_reset = input2_data_ptr;
    }
  }
  else
  {
    // Special case of y4 == 1, in which the innermost loop is a single element
    // and can be combined with the next (y3) as an inner broadcast.
    //
    // Note that this handles the case of pure scalar broadcast when
    // y0 == y1 == y2 == 1. With low overhead it handles cases such as scalar
    // broadcast with batch (as y2 > 1).
    //
    // NOTE The process is the same as the above general case except simplified
    // for y4 == 1 and the loop over y3 is contained within the
    // AddScalarBroadcast function.
    for (int i0 = 0; i0 < y0; ++i0)
    {
      const T *input2_data_ptr = nullptr;
      for (int i1 = 0; i1 < y1; ++i1)
      {
        input2_data_ptr = input2_data_reset;
        for (int i2 = 0; i2 < y2; ++i2)
        {
          scalar_broadcast_f(y3, params, *input1_data_ptr, input2_data_ptr, output_data_ptr);
          input2_data_ptr += y3;
          output_data_ptr += y3;
          input1_data_ptr += 1;
        }
      }
      input2_data_reset = input2_data_ptr;
    }
  }
}

template <typename T>
inline typename std::enable_if_t<is_quant8<T>::value, int32_t>
quant8_sum(const BinaryArithmeticOpParam &params, const T input1_data, const T input2_data)
{
  const int32_t input1_val = params.input1_offset + input1_data;
  const int32_t input2_val = params.input2_offset + input2_data;
  const int32_t shifted_input1_val = input1_val * (1 << params.left_shift);
  const int32_t shifted_input2_val = input2_val * (1 << params.left_shift);
  const int32_t scaled_input1_val = MultiplyByQuantizedMultiplierSmallerThanOneExp(
    shifted_input1_val, params.input1_multiplier, params.input1_shift);
  const int32_t scaled_input2_val = MultiplyByQuantizedMultiplierSmallerThanOneExp(
    shifted_input2_val, params.input2_multiplier, params.input2_shift);
  const int32_t raw_sum = scaled_input1_val + scaled_input2_val;
  const int32_t raw_output = MultiplyByQuantizedMultiplierSmallerThanOneExp(
                               raw_sum, params.output_multiplier, params.output_shift) +
                             params.output_offset;
  const int32_t clamped_output = std::min(params.quantized_activation_max,
                                          std::max(params.quantized_activation_min, raw_output));
  return clamped_output;
}

inline void AddElementwise(int size, const BinaryArithmeticOpParam &params,
                           const uint8_t *input1_data, const uint8_t *input2_data,
                           uint8_t *output_data)
{
  int i = 0;

#ifdef USE_NEON
  const uint8x8_t output_activation_min_vector = vdup_n_u8(params.quantized_activation_min);
  const uint8x8_t output_activation_max_vector = vdup_n_u8(params.quantized_activation_max);
  for (; i <= size - 8; i += 8)
  {
    const uint8x8_t input1_val_original = vld1_u8(input1_data + i);
    const uint8x8_t input2_val_original = vld1_u8(input2_data + i);
    const int16x8_t input1_val_s16 = vreinterpretq_s16_u16(vmovl_u8(input1_val_original));
    const int16x8_t input2_val_s16 = vreinterpretq_s16_u16(vmovl_u8(input2_val_original));
    const int16x8_t input1_val = vaddq_s16(input1_val_s16, vdupq_n_s16(params.input1_offset));
    const int16x8_t input2_val = vaddq_s16(input2_val_s16, vdupq_n_s16(params.input2_offset));
    const int16x4_t input1_val_high = vget_high_s16(input1_val);
    const int16x4_t input1_val_low = vget_low_s16(input1_val);
    const int16x4_t input2_val_high = vget_high_s16(input2_val);
    const int16x4_t input2_val_low = vget_low_s16(input2_val);
    int32x4_t x11 = vmovl_s16(input1_val_low);
    int32x4_t x12 = vmovl_s16(input1_val_high);
    int32x4_t x21 = vmovl_s16(input2_val_low);
    int32x4_t x22 = vmovl_s16(input2_val_high);
    const int32x4_t left_shift_dup = vdupq_n_s32(params.left_shift);
    x11 = vshlq_s32(x11, left_shift_dup);
    x12 = vshlq_s32(x12, left_shift_dup);
    x21 = vshlq_s32(x21, left_shift_dup);
    x22 = vshlq_s32(x22, left_shift_dup);
    x11 = vqrdmulhq_n_s32(x11, params.input1_multiplier);
    x12 = vqrdmulhq_n_s32(x12, params.input1_multiplier);
    x21 = vqrdmulhq_n_s32(x21, params.input2_multiplier);
    x22 = vqrdmulhq_n_s32(x22, params.input2_multiplier);
    const int32x4_t input1_shift_dup = vdupq_n_s32(params.input1_shift);
    const int32x4_t input2_shift_dup = vdupq_n_s32(params.input2_shift);
    x11 = vshlq_s32(x11, input1_shift_dup);
    x12 = vshlq_s32(x12, input1_shift_dup);
    x21 = vshlq_s32(x21, input2_shift_dup);
    x22 = vshlq_s32(x22, input2_shift_dup);
    int32x4_t s1 = vaddq_s32(x11, x21);
    int32x4_t s2 = vaddq_s32(x12, x22);
    s1 = vqrdmulhq_n_s32(s1, params.output_multiplier);
    s2 = vqrdmulhq_n_s32(s2, params.output_multiplier);
    using gemmlowp::RoundingDivideByPOT;
    s1 = RoundingDivideByPOT(s1, -params.output_shift);
    s2 = RoundingDivideByPOT(s2, -params.output_shift);
    const int16x4_t s1_narrowed = vmovn_s32(s1);
    const int16x4_t s2_narrowed = vmovn_s32(s2);
    const int16x8_t s =
      vaddq_s16(vcombine_s16(s1_narrowed, s2_narrowed), vdupq_n_s16(params.output_offset));
    const uint8x8_t clamped =
      vmax_u8(output_activation_min_vector, vmin_u8(output_activation_max_vector, vqmovun_s16(s)));
    vst1_u8(output_data + i, clamped);
  }
#endif // NEON
  for (; i < size; ++i)
  {
    const int32_t input1_val = params.input1_offset + input1_data[i];
    const int32_t input2_val = params.input2_offset + input2_data[i];
    const int32_t shifted_input1_val = input1_val * (1 << params.left_shift);
    const int32_t shifted_input2_val = input2_val * (1 << params.left_shift);
    const int32_t scaled_input1_val = MultiplyByQuantizedMultiplierSmallerThanOneExp(
      shifted_input1_val, params.input1_multiplier, params.input1_shift);
    const int32_t scaled_input2_val = MultiplyByQuantizedMultiplierSmallerThanOneExp(
      shifted_input2_val, params.input2_multiplier, params.input2_shift);
    const int32_t raw_sum = scaled_input1_val + scaled_input2_val;
    const int32_t raw_output = MultiplyByQuantizedMultiplierSmallerThanOneExp(
                                 raw_sum, params.output_multiplier, params.output_shift) +
                               params.output_offset;
    const int32_t clamped_output = std::min(params.quantized_activation_max,
                                            std::max(params.quantized_activation_min, raw_output));
    output_data[i] = static_cast<uint8_t>(clamped_output);
  }
}

inline void AddElementwise(int size, const BinaryArithmeticOpParam &params,
                           const int8_t *input1_data, const int8_t *input2_data,
                           int8_t *output_data)
{
  UNUSED_ALL(size, params, input1_data, input2_data, output_data);
  throw std::runtime_error{"BinaryArithmetic(mul): AddElementwise int8_t is not supported yet"};
}

struct BinaryOpFuncAddFloat
{
#ifdef USE_NEON
  static inline float32x4_t calculate(const float32x4_t &a, const float32x4_t &b)
  {
    return vaddq_f32(a, b);
  }
#endif // USE_NEON
  static inline float calculate(const float a, const float b) { return a + b; }
};

struct BinaryOpFuncSubFloat
{
#ifdef USE_NEON
  static inline float32x4_t calculate(const float32x4_t &a, const float32x4_t &b)
  {
    return vsubq_f32(a, b);
  }
#endif // USE_NEON
  static inline float calculate(const float a, const float b) { return a - b; }
};

struct BinaryOpFuncMulFloat
{
#ifdef USE_NEON
  static inline float32x4_t calculate(const float32x4_t &a, const float32x4_t &b)
  {
    return vmulq_f32(a, b);
  }
#endif // USE_NEON
  static inline float calculate(const float a, const float b) { return a * b; }
};

struct BinaryOpFuncDivFloat
{
#ifdef USE_NEON
#ifdef __aarch64__
  static inline float32x4_t calculate(const float32x4_t &a, const float32x4_t &b)
  {
    return vdivq_f32(a, b);
  }
#endif // __aarch64__
#endif // USE_NEON
  static inline float calculate(const float a, const float b) { return a / b; }
};

template <class BASEOPERATOR> struct BinaryOpFuncSwapArgs
{
  template <typename T> static inline T calculate(const T &a, const T &b)
  {
    return BASEOPERATOR::calculate(b, a);
  }
};

struct BinaryOpActivationFloatNone
{
#ifdef USE_NEON
  static inline float32x4_t applyCeiling(const float32x4_t &value, const float32x4_t &ceilingParam)
  {
    (void)ceilingParam; // suppress unused argument warning
    return value;
  }
  static inline float32x4_t applyFloor(const float32x4_t &value, const float32x4_t &floorParam)
  {
    (void)floorParam;
    return value;
  }
#endif // USE_NEON
  static inline float applyCeiling(const float value, const float ceilingParam)
  {
    (void)ceilingParam;
    return value;
  }
  static inline float applyFloor(const float value, const float floorParam)
  {
    (void)floorParam;
    return value;
  }
};

struct BinaryOpActivationFloatMax
{
#ifdef USE_NEON
  static inline float32x4_t applyCeiling(const float32x4_t &value, const float32x4_t &ceilingParam)
  {
    (void)ceilingParam; // suppress unused argument warning
    return value;
  }
  static inline float32x4_t applyFloor(const float32x4_t &value, const float32x4_t &floorParam)
  {
    return vmaxq_f32(value, floorParam);
  }
#endif // USE_NEON
  static inline float applyCeiling(const float value, const float ceilingParam)
  {
    (void)ceilingParam;
    return value;
  }
  static inline float applyFloor(const float value, const float floorParam)
  {
    return std::max(value, floorParam);
  }
};

struct BinaryOpActivationFloatMinMax
{
#ifdef USE_NEON
  static inline float32x4_t applyCeiling(const float32x4_t &value, const float32x4_t &ceilingParam)
  {
    return vminq_f32(value, ceilingParam);
  }
  static inline float32x4_t applyFloor(const float32x4_t &value, const float32x4_t &floorParam)
  {
    return vmaxq_f32(value, floorParam);
  }
#endif // USE_NEON
  static inline float applyCeiling(const float value, const float ceilingParam)
  {
    return std::min(value, ceilingParam);
  }
  static inline float applyFloor(const float value, const float floorParam)
  {
    return std::max(value, floorParam);
  }
};

template <class OPERATOR, class ACTIVATION>
inline void BinaryOpElementwise(int size, const BinaryArithmeticOpParam &params,
                                const float *input1_data, const float *input2_data,
                                float *output_data)
{
  int i = 0;

#ifdef USE_NEON
  const auto activation_min = vdupq_n_f32(params.float_activation_min);
  const auto activation_max = vdupq_n_f32(params.float_activation_max);
  for (; i <= size - 16; i += 16)
  {
    auto a10 = vld1q_f32(input1_data + i);
    auto a11 = vld1q_f32(input1_data + i + 4);
    auto a12 = vld1q_f32(input1_data + i + 8);
    auto a13 = vld1q_f32(input1_data + i + 12);
    auto a20 = vld1q_f32(input2_data + i);
    auto a21 = vld1q_f32(input2_data + i + 4);
    auto a22 = vld1q_f32(input2_data + i + 8);
    auto a23 = vld1q_f32(input2_data + i + 12);
    auto x0 = OPERATOR::calculate(a10, a20);
    auto x1 = OPERATOR::calculate(a11, a21);
    auto x2 = OPERATOR::calculate(a12, a22);
    auto x3 = OPERATOR::calculate(a13, a23);
    x0 = ACTIVATION::applyFloor(x0, activation_min);
    x1 = ACTIVATION::applyFloor(x1, activation_min);
    x2 = ACTIVATION::applyFloor(x2, activation_min);
    x3 = ACTIVATION::applyFloor(x3, activation_min);
    x0 = ACTIVATION::applyCeiling(x0, activation_max);
    x1 = ACTIVATION::applyCeiling(x1, activation_max);
    x2 = ACTIVATION::applyCeiling(x2, activation_max);
    x3 = ACTIVATION::applyCeiling(x3, activation_max);
    vst1q_f32(output_data + i, x0);
    vst1q_f32(output_data + i + 4, x1);
    vst1q_f32(output_data + i + 8, x2);
    vst1q_f32(output_data + i + 12, x3);
  }
  for (; i <= size - 4; i += 4)
  {
    auto a1 = vld1q_f32(input1_data + i);
    auto a2 = vld1q_f32(input2_data + i);
    auto x = OPERATOR::calculate(a1, a2); // vaddq
    auto x_clamped =
      ACTIVATION::applyCeiling(ACTIVATION::applyFloor(x, activation_min), activation_max);
    vst1q_f32(output_data + i, x_clamped);
  }
#endif // USE_NEON
  for (; i < size; i++)
  {
    auto x = OPERATOR::calculate(input1_data[i], input2_data[i]);
    output_data[i] = ACTIVATION::applyCeiling(
      ACTIVATION::applyFloor(x, params.float_activation_min), params.float_activation_max);
  }
}

// Broadcast binary op template that can often be used for inner loop
// This function will handle scalar_value (LHS) and vector_values (RHS).
// Since it's a float function, input params does not matter here.
template <class OPERATOR, class ACTIVATION>
inline void BinaryOpScalarBroadcast(int size, const BinaryArithmeticOpParam &params,
                                    const float broadcast_value, const float *input2_data,
                                    float *output_data)
{
  int i = 0;

#ifdef USE_NEON
  const auto activation_min = vdupq_n_f32(params.float_activation_min);
  const auto activation_max = vdupq_n_f32(params.float_activation_max);
  const auto broadcast_value_dup = vdupq_n_f32(broadcast_value);
  for (; i <= size - 16; i += 16)
  {
    auto a20 = vld1q_f32(input2_data + i);
    auto a21 = vld1q_f32(input2_data + i + 4);
    auto a22 = vld1q_f32(input2_data + i + 8);
    auto a23 = vld1q_f32(input2_data + i + 12);
    auto x0 = OPERATOR::calculate(broadcast_value_dup, a20);
    auto x1 = OPERATOR::calculate(broadcast_value_dup, a21);
    auto x2 = OPERATOR::calculate(broadcast_value_dup, a22);
    auto x3 = OPERATOR::calculate(broadcast_value_dup, a23);
    x0 = ACTIVATION::applyFloor(x0, activation_min);
    x1 = ACTIVATION::applyFloor(x1, activation_min);
    x2 = ACTIVATION::applyFloor(x2, activation_min);
    x3 = ACTIVATION::applyFloor(x3, activation_min);
    x0 = ACTIVATION::applyCeiling(x0, activation_max);
    x1 = ACTIVATION::applyCeiling(x1, activation_max);
    x2 = ACTIVATION::applyCeiling(x2, activation_max);
    x3 = ACTIVATION::applyCeiling(x3, activation_max);
    vst1q_f32(output_data + i, x0);
    vst1q_f32(output_data + i + 4, x1);
    vst1q_f32(output_data + i + 8, x2);
    vst1q_f32(output_data + i + 12, x3);
  }
  for (; i <= size - 4; i += 4)
  {
    auto a2 = vld1q_f32(input2_data + i);
    auto x = OPERATOR::calculate(broadcast_value_dup, a2);
    auto x_clamped =
      ACTIVATION::applyCeiling(ACTIVATION::applyFloor(x, activation_min), activation_max);
    vst1q_f32(output_data + i, x_clamped);
  }
#endif // USE_NEON
  for (; i < size; i++)
  {
    auto x = OPERATOR::calculate(broadcast_value, input2_data[i]);
    output_data[i] = ACTIVATION::applyCeiling(
      ACTIVATION::applyFloor(x, params.float_activation_min), params.float_activation_max);
  }
}

using BinaryOpImplFloatFuncs =
  std::pair<void (*)(int, const BinaryArithmeticOpParam &, const float *, const float *, float *),
            void (*)(int, const BinaryArithmeticOpParam &, const float, const float *, float *)>;

template <class FUNC>
inline BinaryOpImplFloatFuncs
getBinaryOpWithActivationImplFloat(const BinaryArithmeticOpParam &params)
{
  if (params.float_activation_max == std::numeric_limits<float>::max())
    if (params.float_activation_min == std::numeric_limits<float>::lowest())
      return BinaryOpImplFloatFuncs(BinaryOpElementwise<FUNC, BinaryOpActivationFloatNone>,
                                    BinaryOpScalarBroadcast<FUNC, BinaryOpActivationFloatNone>);
    else
      return BinaryOpImplFloatFuncs(BinaryOpElementwise<FUNC, BinaryOpActivationFloatMax>,
                                    BinaryOpScalarBroadcast<FUNC, BinaryOpActivationFloatMax>);
  else
    return BinaryOpImplFloatFuncs(BinaryOpElementwise<FUNC, BinaryOpActivationFloatMinMax>,
                                  BinaryOpScalarBroadcast<FUNC, BinaryOpActivationFloatMinMax>);
}

template <typename T>
inline typename std::enable_if_t<is_quant8<T>::value>
Add(const BinaryArithmeticOpParam &params, const Shape &input1_shape, const T *input1_data,
    const Shape &input2_shape, const T *input2_data, const Shape &output_shape, T *output_data)
{
  const int flat_size = MatchingElementsSize(input1_shape, input2_shape, output_shape);
  AddElementwise(flat_size, params, input1_data, input2_data, output_data);
}

inline void Add(const BinaryArithmeticOpParam &params, const Shape &input1_shape,
                const float *input1_data, const Shape &input2_shape, const float *input2_data,
                const Shape &output_shape, float *output_data)
{
  const int flat_size = MatchingElementsSize(input1_shape, input2_shape, output_shape);
  auto implFuncs = getBinaryOpWithActivationImplFloat<BinaryOpFuncAddFloat>(params);
  (*implFuncs.first)(flat_size, params, input1_data, input2_data, output_data);
}

// Scalar-broadcast add that can be used for inner loop of more general
// broadcast add, so that, for example, scalar-broadcast with batch will still
// be fast.
template <typename T>
inline typename std::enable_if_t<is_quant8<T>::value>
AddScalarBroadcast(int size, const BinaryArithmeticOpParam &params, T broadcast_value,
                   const T *input2_data, T *output_data)
{
  int i = 0;
  int32_t clamped_output;
  for (; i < size; ++i)
  {
    clamped_output = quant8_sum(params, broadcast_value, input2_data[i]);
    output_data[i] = static_cast<T>(clamped_output);
  }
}

template <typename T>
inline typename std::enable_if_t<is_quant8<T>::value>
BroadcastAddDispatch(const BinaryArithmeticOpParam &params, const Shape &input1_shape,
                     const T *input1_data, const Shape &input2_shape, const T *input2_data,
                     const Shape &output_shape, T *output_data)
{
  if (params.broadcast_category == BroadcastableOpCategory::kGenericBroadcast)
  {
    const std::function<T(const BinaryArithmeticOpParam &, const T &, const T &)> fn =
      [](const BinaryArithmeticOpParam &params, const T &a, const T &b) {
        return static_cast<T>(quant8_sum(params, a, b));
      };
    reference::BroadcastBinaryArithmeticOpSlow(params, input1_shape, input1_data, input2_shape,
                                               input2_data, output_shape, output_data, fn);
  }
  else
  {
    BinaryBroadcastFiveFold(
      params, params.broadcast_category == BroadcastableOpCategory::kSecondInputBroadcastsFast,
      input1_shape, input1_data, input2_shape, input2_data, output_shape, output_data,
      static_cast<void (*)(int, const BinaryArithmeticOpParam &, const T *, const T *, T *)>(
        AddElementwise),
      static_cast<void (*)(int, const BinaryArithmeticOpParam &, T, const T *, T *)>(
        AddScalarBroadcast));
  }
}

inline void BroadcastAddDispatch(const BinaryArithmeticOpParam &params, const Shape &input1_shape,
                                 const float *input1_data, const Shape &input2_shape,
                                 const float *input2_data, const Shape &output_shape,
                                 float *output_data)
{
  if (params.broadcast_category == BroadcastableOpCategory::kGenericBroadcast)
  {
    const std::function<float(const float &, const float &)> fn =
      [](const float &a, const float &b) -> float { return a + b; };
    reference::BroadcastBinaryArithmeticOpSlow(params, input1_shape, input1_data, input2_shape,
                                               input2_data, output_shape, output_data, fn);
  }
  else
  {
    auto implFuncs = getBinaryOpWithActivationImplFloat<BinaryOpFuncAddFloat>(params);

    BinaryBroadcastFiveFold(
      params, params.broadcast_category == BroadcastableOpCategory::kSecondInputBroadcastsFast,
      input1_shape, input1_data, input2_shape, input2_data, output_shape, output_data,
      implFuncs.first, implFuncs.second);
  }
}

inline void Sub(const BinaryArithmeticOpParam &params, const Shape &input1_shape,
                const float *input1_data, const Shape &input2_shape, const float *input2_data,
                const Shape &output_shape, float *output_data)
{
  const int flat_size = MatchingElementsSize(input1_shape, input2_shape, output_shape);
  auto implFuncs = getBinaryOpWithActivationImplFloat<BinaryOpFuncSubFloat>(params);
  (*implFuncs.first)(flat_size, params, input1_data, input2_data, output_data);
}

inline void BroadcastSubDispatch(const BinaryArithmeticOpParam &params, const Shape &input1_shape,
                                 const float *input1_data, const Shape &input2_shape,
                                 const float *input2_data, const Shape &output_shape,
                                 float *output_data)
{
  if (params.broadcast_category == BroadcastableOpCategory::kFirstInputBroadcastsFast)
  {
    auto implFuncs = getBinaryOpWithActivationImplFloat<BinaryOpFuncSubFloat>(params);
    BinaryBroadcastFiveFold(params, false, input1_shape, input1_data, input2_shape, input2_data,
                            output_shape, output_data, implFuncs.first, implFuncs.second);
  }
  else if (params.broadcast_category == BroadcastableOpCategory::kSecondInputBroadcastsFast)
  {
    auto implFuncs =
      getBinaryOpWithActivationImplFloat<BinaryOpFuncSwapArgs<BinaryOpFuncSubFloat>>(params);
    BinaryBroadcastFiveFold(params, true, input1_shape, input1_data, input2_shape, input2_data,
                            output_shape, output_data, implFuncs.first, implFuncs.second);
  }
  else
  {
    const std::function<float(const float &, const float &)> fn =
      [](const float &a, const float &b) -> float { return a - b; };
    reference::BroadcastBinaryArithmeticOpSlow(params, input1_shape, input1_data, input2_shape,
                                               input2_data, output_shape, output_data, fn);
  }
}

template <typename T>
inline typename std::enable_if_t<is_quant8<T>::value, int32_t>
quant8_mul(const BinaryArithmeticOpParam &params, const T input1_data, const T input2_data)
{
  const int32_t input1_val = params.input1_offset + input1_data;
  const int32_t input2_val = params.input2_offset + input2_data;
  const int32_t unclamped_result =
    params.output_offset + MultiplyByQuantizedMultiplier(input1_val * input2_val,
                                                         params.output_multiplier,
                                                         params.output_shift);
  const int32_t clamped_output = std::min(
    params.quantized_activation_max, std::max(params.quantized_activation_min, unclamped_result));

  return clamped_output;
}

inline void MulElementwise(int size, const BinaryArithmeticOpParam &params,
                           const uint8_t *input1_data, const uint8_t *input2_data,
                           uint8_t *output_data)
{
  int i = 0;

#ifdef USE_NEON
  const auto input1_offset_vector = vdupq_n_s16(params.input1_offset);
  const auto input2_offset_vector = vdupq_n_s16(params.input2_offset);
  const auto output_offset_vector = vdupq_n_s16(params.output_offset);
  const auto output_activation_min_vector = vdup_n_u8(params.quantized_activation_min);
  const auto output_activation_max_vector = vdup_n_u8(params.quantized_activation_max);
  const int left_shift = std::max(0, params.output_shift);
  const int right_shift = std::max(0, -params.output_shift);
  const int32x4_t left_shift_vec = vdupq_n_s32(left_shift);
  for (; i <= size - 8; i += 8)
  {
    // We load / store 8 at a time, multiplying as two sets of 4 int32s.
    const auto input1_val_original = vld1_u8(input1_data + i);
    const auto input2_val_original = vld1_u8(input2_data + i);
    const auto input1_val_s16 = vreinterpretq_s16_u16(vmovl_u8(input1_val_original));
    const auto input2_val_s16 = vreinterpretq_s16_u16(vmovl_u8(input2_val_original));
    const auto input1_val = vaddq_s16(input1_val_s16, input1_offset_vector);
    const auto input2_val = vaddq_s16(input2_val_s16, input2_offset_vector);

    const auto input1_val_low = vget_low_s16(input1_val);
    const auto input1_val_high = vget_high_s16(input1_val);
    const auto input2_val_low = vget_low_s16(input2_val);
    const auto input2_val_high = vget_high_s16(input2_val);

    auto p1 = vmull_s16(input2_val_low, input1_val_low);
    auto p2 = vmull_s16(input2_val_high, input1_val_high);

    p1 = vshlq_s32(p1, left_shift_vec);
    p2 = vshlq_s32(p2, left_shift_vec);
    p1 = vqrdmulhq_n_s32(p1, params.output_multiplier);
    p2 = vqrdmulhq_n_s32(p2, params.output_multiplier);
    using gemmlowp::RoundingDivideByPOT;
    p1 = RoundingDivideByPOT(p1, right_shift);
    p2 = RoundingDivideByPOT(p2, right_shift);

    const auto p1_narrowed = vqmovn_s32(p1);
    const auto p2_narrowed = vqmovn_s32(p2);
    const auto p = vaddq_s16(vcombine_s16(p1_narrowed, p2_narrowed), output_offset_vector);
    const auto clamped =
      vmax_u8(output_activation_min_vector, vmin_u8(output_activation_max_vector, vqmovun_s16(p)));
    vst1_u8(output_data + i, clamped);
  }
#endif // NEON

  for (; i < size; ++i)
  {
    const int32_t input1_val = params.input1_offset + input1_data[i];
    const int32_t input2_val = params.input2_offset + input2_data[i];
    const int32_t unclamped_result =
      params.output_offset + MultiplyByQuantizedMultiplier(input1_val * input2_val,
                                                           params.output_multiplier,
                                                           params.output_shift);
    const int32_t clamped_output = std::min(
      params.quantized_activation_max, std::max(params.quantized_activation_min, unclamped_result));
    output_data[i] = static_cast<uint8_t>(clamped_output);
  }
}

template <typename T>
inline typename std::enable_if_t<is_quant8<T>::value>
Mul(const BinaryArithmeticOpParam &params, const Shape &input1_shape, const T *input1_data,
    const Shape &input2_shape, const T *input2_data, const Shape &output_shape, T *output_data)
{
  const int flat_size = MatchingElementsSize(input1_shape, input2_shape, output_shape);
  MulElementwise(flat_size, params, input1_data, input2_data, output_data);
}

inline void Mul(const BinaryArithmeticOpParam &params, const Shape &input1_shape,
                const float *input1_data, const Shape &input2_shape, const float *input2_data,
                const Shape &output_shape, float *output_data)
{
  const int flat_size = MatchingElementsSize(input1_shape, input2_shape, output_shape);
  auto implFuncs = getBinaryOpWithActivationImplFloat<BinaryOpFuncMulFloat>(params);
  (*implFuncs.first)(flat_size, params, input1_data, input2_data, output_data);
}

template <typename T>
inline typename std::enable_if_t<is_quant8<T>::value>
MulSimpleBroadcast(int size, const BinaryArithmeticOpParam &params, const T broadcast_value,
                   const T *input2_data, T *output_data)
{
  int i = 0;
  int32_t clamped_output;
  for (; i < size; ++i)
  {
    clamped_output = quant8_mul(params, broadcast_value, input2_data[i]);
    output_data[i] = static_cast<T>(clamped_output);
  }
}

template <typename T>
inline typename std::enable_if_t<is_quant8<T>::value>
BroadcastMulDispatch(const BinaryArithmeticOpParam &params, const Shape &input1_shape,
                     const T *input1_data, const Shape &input2_shape, const T *input2_data,
                     const Shape &output_shape, T *output_data)
{
  if (params.broadcast_category == BroadcastableOpCategory::kGenericBroadcast)
  {
    const std::function<T(const BinaryArithmeticOpParam &, const T &, const T &)> fn =
      [](const BinaryArithmeticOpParam &params, const T &a, const T &b) {
        return static_cast<T>(quant8_mul(params, a, b));
      };
    reference::BroadcastBinaryArithmeticOpSlow(params, input1_shape, input1_data, input2_shape,
                                               input2_data, output_shape, output_data, fn);
    return;
  }
  BinaryBroadcastFiveFold(
    params, params.broadcast_category == BroadcastableOpCategory::kSecondInputBroadcastsFast,
    input1_shape, input1_data, input2_shape, input2_data, output_shape, output_data,
    static_cast<void (*)(int, const BinaryArithmeticOpParam &, const T *, const T *, T *)>(
      MulElementwise),
    static_cast<void (*)(int, const BinaryArithmeticOpParam &, T, const T *, T *)>(
      MulSimpleBroadcast));
}

inline void BroadcastMulDispatch(const BinaryArithmeticOpParam &params, const Shape &input1_shape,
                                 const float *input1_data, const Shape &input2_shape,
                                 const float *input2_data, const Shape &output_shape,
                                 float *output_data)
{
  if (params.broadcast_category == BroadcastableOpCategory::kGenericBroadcast)
  {
    // TODO: Use GetBinaryArithmeticFn
    const std::function<float(const float &, const float &)> fn =
      [](const float &a, const float &b) -> float { return a * b; };
    reference::BroadcastBinaryArithmeticOpSlow(params, input1_shape, input1_data, input2_shape,
                                               input2_data, output_shape, output_data, fn);
    return;
  }
  auto implFuncs = getBinaryOpWithActivationImplFloat<BinaryOpFuncMulFloat>(params);
  BinaryBroadcastFiveFold(
    params, params.broadcast_category == BroadcastableOpCategory::kSecondInputBroadcastsFast,
    input1_shape, input1_data, input2_shape, input2_data, output_shape, output_data,
    implFuncs.first, implFuncs.second);
}

inline void Div(const BinaryArithmeticOpParam &params, const Shape &input1_shape,
                const float *input1_data, const Shape &input2_shape, const float *input2_data,
                const Shape &output_shape, float *output_data)
{
#ifdef __aarch64__
  const int flat_size = MatchingElementsSize(input1_shape, input2_shape, output_shape);
  auto implFuncs = getBinaryOpWithActivationImplFloat<BinaryOpFuncDivFloat>(params);
  (*implFuncs.first)(flat_size, params, input1_data, input2_data, output_data);
#else
  const std::function<float(const float &, const float &)> fn =
    [](const float &a, const float &b) -> float { return a / b; };
  reference::BinaryArithmeticOp(params, input1_shape, input1_data, input2_shape, input2_data,
                                output_shape, output_data, fn);
#endif // __aarch64__
}

inline void BroadcastDivDispatch(const BinaryArithmeticOpParam &params, const Shape &input1_shape,
                                 const float *input1_data, const Shape &input2_shape,
                                 const float *input2_data, const Shape &output_shape,
                                 float *output_data)
{
#ifdef __aarch64__
  if (params.broadcast_category == BroadcastableOpCategory::kFirstInputBroadcastsFast)
  {
    auto implFuncs = getBinaryOpWithActivationImplFloat<BinaryOpFuncDivFloat>(params);
    BinaryBroadcastFiveFold(params, false, input1_shape, input1_data, input2_shape, input2_data,
                            output_shape, output_data, implFuncs.first, implFuncs.second);
  }
  else if (params.broadcast_category == BroadcastableOpCategory::kSecondInputBroadcastsFast)
  {
    auto implFuncs =
      getBinaryOpWithActivationImplFloat<BinaryOpFuncSwapArgs<BinaryOpFuncDivFloat>>(params);
    BinaryBroadcastFiveFold(params, true, input1_shape, input1_data, input2_shape, input2_data,
                            output_shape, output_data, implFuncs.first, implFuncs.second);
  }
  else
#endif // __aarch64__
  {
    const std::function<float(const float &, const float &)> fn =
      [](const float &a, const float &b) -> float { return a / b; };
    reference::BroadcastBinaryArithmeticOpSlow(params, input1_shape, input1_data, input2_shape,
                                               input2_data, output_shape, output_data, fn);
  }
}

} // namespace optimized
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_OPTIMIZED_BINARYARITHMETICOPS_H__
