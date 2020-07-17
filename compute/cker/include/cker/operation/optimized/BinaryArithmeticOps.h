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
inline void BinaryBroadcastFiveFold(const BinaryArithmeticOpParam &params,
                                    const Shape & /* unswitched_input1_shape */,
                                    const T *unswitched_input1_data,
                                    const Shape & /* unswitched_input2_shape */,
                                    const T *unswitched_input2_data,
                                    const Shape & /* output_shape */, T *output_data,
                                    ElementwiseF elementwise_f, ScalarBroadcastF scalar_broadcast_f)
{
  const bool use_unswitched =
      params.broadcast_category == BroadcastableOpCategory::kFirstInputBroadcastsFast;

  const T *input1_data = use_unswitched ? unswitched_input1_data : unswitched_input2_data;
  const T *input2_data = use_unswitched ? unswitched_input2_data : unswitched_input1_data;

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

inline int32_t quant8_sum(const BinaryArithmeticOpParam &params, const uint8_t input1_data,
                          const uint8_t input2_data)
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

inline void AddElementwiseQuant8(int size, const BinaryArithmeticOpParam &params,
                                 const uint8_t *input1_data, const uint8_t *input2_data,
                                 uint8_t *output_data)
{
  int i = 0;
  for (; i < size; ++i)
  {
    int32_t clamped_output = quant8_sum(params, input1_data[i], input2_data[i]);
    output_data[i] = static_cast<uint8_t>(clamped_output);
  }
}

inline void AddElementwise(int size, const BinaryArithmeticOpParam &params,
                           const float *input1_data, const float *input2_data, float *output_data)
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
    auto x0 = vaddq_f32(a10, a20);
    auto x1 = vaddq_f32(a11, a21);
    auto x2 = vaddq_f32(a12, a22);
    auto x3 = vaddq_f32(a13, a23);
    x0 = vmaxq_f32(activation_min, x0);
    x1 = vmaxq_f32(activation_min, x1);
    x2 = vmaxq_f32(activation_min, x2);
    x3 = vmaxq_f32(activation_min, x3);
    x0 = vminq_f32(activation_max, x0);
    x1 = vminq_f32(activation_max, x1);
    x2 = vminq_f32(activation_max, x2);
    x3 = vminq_f32(activation_max, x3);
    vst1q_f32(output_data + i, x0);
    vst1q_f32(output_data + i + 4, x1);
    vst1q_f32(output_data + i + 8, x2);
    vst1q_f32(output_data + i + 12, x3);
  }
  for (; i <= size - 4; i += 4)
  {
    auto a1 = vld1q_f32(input1_data + i);
    auto a2 = vld1q_f32(input2_data + i);
    auto x = vaddq_f32(a1, a2);
    x = vmaxq_f32(activation_min, x);
    x = vminq_f32(activation_max, x);
    vst1q_f32(output_data + i, x);
  }
#endif // NEON
  for (; i < size; i++)
  {
    auto x = input1_data[i] + input2_data[i];
    output_data[i] = ActivationFunctionWithMinMax<float>(x, params.float_activation_min,
                                                         params.float_activation_max);
  }
}

inline void AddQuant8(const BinaryArithmeticOpParam &params, const Shape &input1_shape,
                      const uint8_t *input1_data, const Shape &input2_shape,
                      const uint8_t *input2_data, const Shape &output_shape, uint8_t *output_data)
{
  const int flat_size = MatchingElementsSize(input1_shape, input2_shape, output_shape);
  AddElementwiseQuant8(flat_size, params, input1_data, input2_data, output_data);
}

inline void Add(const BinaryArithmeticOpParam &params, const Shape &input1_shape,
                const float *input1_data, const Shape &input2_shape, const float *input2_data,
                const Shape &output_shape, float *output_data)
{
  const int flat_size = MatchingElementsSize(input1_shape, input2_shape, output_shape);
  AddElementwise(flat_size, params, input1_data, input2_data, output_data);
}

// Scalar-broadcast add that can be used for inner loop of more general
// broadcast add, so that, for example, scalar-broadcast with batch will still
// be fast.
inline void AddScalarBroadcastQuant8(int size, const BinaryArithmeticOpParam &params,
                                     uint8_t broadcast_value, const uint8_t *input2_data,
                                     uint8_t *output_data)
{
  int i = 0;
  int32_t clamped_output;
  for (; i < size; ++i)
  {
    clamped_output = quant8_sum(params, broadcast_value, input2_data[i]);
    output_data[i] = static_cast<uint8_t>(clamped_output);
  }
}

inline void AddScalarBroadcast(int size, const BinaryArithmeticOpParam &params,
                               float broadcast_value, const float *input2_data, float *output_data)
{
  int i = 0;
#ifdef USE_NEON
  const float32x4_t output_activation_min_vector = vdupq_n_f32(params.float_activation_min);
  const float32x4_t output_activation_max_vector = vdupq_n_f32(params.float_activation_max);
  const float32x4_t broadcast_value_dup = vdupq_n_f32(broadcast_value);
  for (; i <= size - 4; i += 4)
  {
    const float32x4_t input2_val_original = vld1q_f32(input2_data + i);

    const float32x4_t output = vaddq_f32(input2_val_original, broadcast_value_dup);

    const float32x4_t clamped =
        vmaxq_f32(output_activation_min_vector, vminq_f32(output_activation_max_vector, output));
    vst1q_f32(output_data + i, clamped);
  }
#endif // NEON
  for (; i < size; ++i)
  {
    auto x = broadcast_value + input2_data[i];
    output_data[i] = ActivationFunctionWithMinMax<float>(x, params.float_activation_min,
                                                         params.float_activation_max);
  }
}

inline void BroadcastAddDispatchQuant8(const BinaryArithmeticOpParam &params,
                                       const Shape &input1_shape, const uint8_t *input1_data,
                                       const Shape &input2_shape, const uint8_t *input2_data,
                                       const Shape &output_shape, uint8_t *output_data)
{
  if (params.broadcast_category == BroadcastableOpCategory::kGenericBroadcast)
  {
    const std::function<uint8_t(const BinaryArithmeticOpParam &, const uint8_t &, const uint8_t &)>
        fn = [](const BinaryArithmeticOpParam &params, const uint8_t &a,
                const uint8_t &b) -> uint8_t {
      return static_cast<uint8_t>(quant8_sum(params, a, b));
    };
    reference::BroadcastBinaryArithmeticOpSlowQuant8(params, input1_shape, input1_data,
                                                     input2_shape, input2_data, output_shape,
                                                     output_data, fn);
  }
  else
  {
    BinaryBroadcastFiveFold(
        params, input1_shape, input1_data, input2_shape, input2_data, output_shape, output_data,
        static_cast<void (*)(int, const BinaryArithmeticOpParam &, const uint8_t *, const uint8_t *,
                             uint8_t *)>(AddElementwiseQuant8),
        static_cast<void (*)(int, const BinaryArithmeticOpParam &, uint8_t, const uint8_t *,
                             uint8_t *)>(AddScalarBroadcastQuant8));
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
    BinaryBroadcastFiveFold(
        params, input1_shape, input1_data, input2_shape, input2_data, output_shape, output_data,
        static_cast<void (*)(int, const BinaryArithmeticOpParam &, const float *, const float *,
                             float *)>(AddElementwise),
        static_cast<void (*)(int, const BinaryArithmeticOpParam &, float, const float *, float *)>(
            AddScalarBroadcast));
  }
}

inline void Sub(const BinaryArithmeticOpParam &params, const Shape &input1_shape,
                const float *input1_data, const Shape &input2_shape, const float *input2_data,
                const Shape &output_shape, float *output_data)
{
  int i = 0;
  const int size = MatchingElementsSize(input1_shape, input2_shape, output_shape);
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
    auto x0 = vsubq_f32(a10, a20);
    auto x1 = vsubq_f32(a11, a21);
    auto x2 = vsubq_f32(a12, a22);
    auto x3 = vsubq_f32(a13, a23);
    x0 = vmaxq_f32(activation_min, x0);
    x1 = vmaxq_f32(activation_min, x1);
    x2 = vmaxq_f32(activation_min, x2);
    x3 = vmaxq_f32(activation_min, x3);
    x0 = vminq_f32(activation_max, x0);
    x1 = vminq_f32(activation_max, x1);
    x2 = vminq_f32(activation_max, x2);
    x3 = vminq_f32(activation_max, x3);
    vst1q_f32(output_data + i, x0);
    vst1q_f32(output_data + i + 4, x1);
    vst1q_f32(output_data + i + 8, x2);
    vst1q_f32(output_data + i + 12, x3);
  }
  for (; i <= size - 4; i += 4)
  {
    auto a1 = vld1q_f32(input1_data + i);
    auto a2 = vld1q_f32(input2_data + i);
    auto x = vsubq_f32(a1, a2);
    x = vmaxq_f32(activation_min, x);
    x = vminq_f32(activation_max, x);
    vst1q_f32(output_data + i, x);
  }
#endif // NEON

  for (; i < size; i++)
  {
    auto x = input1_data[i] - input2_data[i];
    output_data[i] =
        ActivationFunctionWithMinMax(x, params.float_activation_min, params.float_activation_max);
  }
}

inline int32_t quant8_mul(const BinaryArithmeticOpParam &params, const uint8_t input1_data,
                          const uint8_t input2_data)
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

inline void MulElementwiseQuant8(int size, const BinaryArithmeticOpParam &params,
                                 const uint8_t *input1_data, const uint8_t *input2_data,
                                 uint8_t *output_data)
{
  int i = 0;
  int32_t clamped_output;
  for (; i < size; i++)
  {
    clamped_output = quant8_mul(params, input1_data[i], input2_data[i]);
    output_data[i] = static_cast<uint8_t>(clamped_output);
  }
}

inline void MulElementwise(int size, const BinaryArithmeticOpParam &params,
                           const float *input1_data, const float *input2_data, float *output_data)
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
    auto x0 = vmulq_f32(a10, a20);
    auto x1 = vmulq_f32(a11, a21);
    auto x2 = vmulq_f32(a12, a22);
    auto x3 = vmulq_f32(a13, a23);
    x0 = vmaxq_f32(activation_min, x0);
    x1 = vmaxq_f32(activation_min, x1);
    x2 = vmaxq_f32(activation_min, x2);
    x3 = vmaxq_f32(activation_min, x3);
    x0 = vminq_f32(activation_max, x0);
    x1 = vminq_f32(activation_max, x1);
    x2 = vminq_f32(activation_max, x2);
    x3 = vminq_f32(activation_max, x3);
    vst1q_f32(output_data + i, x0);
    vst1q_f32(output_data + i + 4, x1);
    vst1q_f32(output_data + i + 8, x2);
    vst1q_f32(output_data + i + 12, x3);
  }
  for (; i <= size - 4; i += 4)
  {
    auto a1 = vld1q_f32(input1_data + i);
    auto a2 = vld1q_f32(input2_data + i);
    auto x = vmulq_f32(a1, a2);
    x = vmaxq_f32(activation_min, x);
    x = vminq_f32(activation_max, x);
    vst1q_f32(output_data + i, x);
  }
#endif // NEON

  for (; i < size; i++)
  {
    auto x = input1_data[i] * input2_data[i];
    output_data[i] =
        ActivationFunctionWithMinMax(x, params.float_activation_min, params.float_activation_max);
  }
}

inline void MulQuant8(const BinaryArithmeticOpParam &params, const Shape &input1_shape,
                      const uint8_t *input1_data, const Shape &input2_shape,
                      const uint8_t *input2_data, const Shape &output_shape, uint8_t *output_data)
{
  const int flat_size = MatchingElementsSize(input1_shape, input2_shape, output_shape);
  MulElementwiseQuant8(flat_size, params, input1_data, input2_data, output_data);
}

inline void Mul(const BinaryArithmeticOpParam &params, const Shape &input1_shape,
                const float *input1_data, const Shape &input2_shape, const float *input2_data,
                const Shape &output_shape, float *output_data)
{
  const int flat_size = MatchingElementsSize(input1_shape, input2_shape, output_shape);
  MulElementwise(flat_size, params, input1_data, input2_data, output_data);
}

inline void MulSimpleBroadcastQuant8(int size, const BinaryArithmeticOpParam &params,
                                     const uint8_t broadcast_value, const uint8_t *input2_data,
                                     uint8_t *output_data)
{
  int i = 0;
  int32_t clamped_output;
  for (; i < size; ++i)
  {
    clamped_output = quant8_mul(params, broadcast_value, input2_data[i]);
    output_data[i] = static_cast<uint8_t>(clamped_output);
  }
}

// Broadcast mul that can often be used for inner loop of broadcast Mul.
// This function will handle scalar_value (LHS) * vector_values (RHS).
// Since it's a float function, input params does not matter here.
inline void MulSimpleBroadcast(int size, const BinaryArithmeticOpParam &params,
                               const float broadcast_value, const float *input2_data,
                               float *output_data)
{
  int i = 0;
#ifdef USE_NEON
  const float32x4_t output_activation_min_vector = vdupq_n_f32(params.float_activation_min);
  const float32x4_t output_activation_max_vector = vdupq_n_f32(params.float_activation_max);
  const float32x4_t broadcast_value_dup = vdupq_n_f32(broadcast_value);
  for (; i <= size - 4; i += 4)
  {
    const float32x4_t input2_val_original = vld1q_f32(input2_data + i);

    const float32x4_t output = vmulq_f32(input2_val_original, broadcast_value_dup);

    const float32x4_t clamped =
        vmaxq_f32(output_activation_min_vector, vminq_f32(output_activation_max_vector, output));
    vst1q_f32(output_data + i, clamped);
  }
#endif // NEON

  for (; i < size; ++i)
  {
    float x = broadcast_value * input2_data[i];
    output_data[i] =
        ActivationFunctionWithMinMax(x, params.float_activation_min, params.float_activation_max);
  }
}

inline void BroadcastMulDispatchQuant8(const BinaryArithmeticOpParam &params,
                                       const Shape &input1_shape, const uint8_t *input1_data,
                                       const Shape &input2_shape, const uint8_t *input2_data,
                                       const Shape &output_shape, uint8_t *output_data)
{
  if (params.broadcast_category == BroadcastableOpCategory::kGenericBroadcast)
  {
    const std::function<uint8_t(const BinaryArithmeticOpParam &, const uint8_t &, const uint8_t &)>
        fn = [](const BinaryArithmeticOpParam &params, const uint8_t &a,
                const uint8_t &b) -> uint8_t {
      return static_cast<uint8_t>(quant8_mul(params, a, b));
    };
    reference::BroadcastBinaryArithmeticOpSlowQuant8(params, input1_shape, input1_data,
                                                     input2_shape, input2_data, output_shape,
                                                     output_data, fn);
    return;
  }
  BinaryBroadcastFiveFold(
      params, input1_shape, input1_data, input2_shape, input2_data, output_shape, output_data,
      static_cast<void (*)(int, const BinaryArithmeticOpParam &, const uint8_t *, const uint8_t *,
                           uint8_t *)>(MulElementwiseQuant8),
      static_cast<void (*)(int, const BinaryArithmeticOpParam &, uint8_t, const uint8_t *,
                           uint8_t *)>(MulSimpleBroadcastQuant8));
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
  BinaryBroadcastFiveFold(
      params, input1_shape, input1_data, input2_shape, input2_data, output_shape, output_data,
      static_cast<void (*)(int, const BinaryArithmeticOpParam &, const float *, const float *,
                           float *)>(MulElementwise),
      static_cast<void (*)(int, const BinaryArithmeticOpParam &, float, const float *, float *)>(
          MulSimpleBroadcast));
}

} // namespace optimized
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_OPTIMIZED_BINARYARITHMETICOPS_H__
