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

namespace nnfw
{
namespace cker
{
namespace optimized
{

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
    output_data[i] =
        ActivationFunctionWithMinMax(x, params.float_activation_min, params.float_activation_max);
  }
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
    output_data[i] =
        ActivationFunctionWithMinMax(x, params.float_activation_min, params.float_activation_max);
  }
}

inline void BroadcastAddFivefold(const BinaryArithmeticOpParam &params,
                                 const Shape & /* unswitched_input1_shape */,
                                 const float *unswitched_input1_data,
                                 const Shape & /* unswitched_input2_shape */,
                                 const float *unswitched_input2_data,
                                 const Shape & /* output_shape */, float *output_data)
{
  const bool use_unswitched =
      params.broadcast_category == BroadcastableOpCategory::kFirstInputBroadcastsFast;

  const float *input1_data = use_unswitched ? unswitched_input1_data : unswitched_input2_data;
  const float *input2_data = use_unswitched ? unswitched_input2_data : unswitched_input1_data;

  // Fivefold nested loops. The second input resets its position for each
  // iteration of the second loop. The first input resets its position at the
  // beginning of the fourth loop. The innermost loop is an elementwise add of
  // sections of the arrays.
  float *output_data_ptr = output_data;
  const float *input1_data_ptr = input1_data;
  const float *input2_data_reset = input2_data;
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
      const float *input2_data_ptr = nullptr;
      for (int i1 = 0; i1 < y1; ++i1)
      {
        input2_data_ptr = input2_data_reset;
        for (int i2 = 0; i2 < y2; ++i2)
        {
          for (int i3 = 0; i3 < y3; ++i3)
          {
            AddElementwise(y4, params, input1_data_ptr, input2_data_ptr, output_data_ptr);
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
      const float *input2_data_ptr = nullptr;
      for (int i1 = 0; i1 < y1; ++i1)
      {
        input2_data_ptr = input2_data_reset;
        for (int i2 = 0; i2 < y2; ++i2)
        {
          AddScalarBroadcast(y3, params, *input1_data_ptr, input2_data_ptr, output_data_ptr);
          input2_data_ptr += y3;
          output_data_ptr += y3;
          input1_data_ptr += 1;
        }
      }
      input2_data_reset = input2_data_ptr;
    }
  }
}

inline void BroadcastAddDispatch(const BinaryArithmeticOpParam &params, const Shape &input1_shape,
                                 const float *input1_data, const Shape &input2_shape,
                                 const float *input2_data, const Shape &output_shape,
                                 float *output_data)
{
  if (params.broadcast_category == BroadcastableOpCategory::kGenericBroadcast)
  {
    // TODO: Use GetBinaryArithmeticFn
    const std::function<float(const float &, const float &)> fn =
        [](const float &a, const float &b) -> float { return a + b; };
    reference::BroadcastBinaryArithmeticOpSlow(params, input1_shape, input1_data, input2_shape,
                                               input2_data, output_shape, output_data, fn);
    return;
  }
  BroadcastAddFivefold(params, input1_shape, input1_data, input2_shape, input2_data, output_shape,
                       output_data);
}

inline void Sub(const BinaryArithmeticOpParam &params, const Shape &input1_shape,
                const float *input1_data, const Shape &input2_shape, const float *input2_data,
                const Shape &output_shape, float *output_data)
{
  int i = 0;
  const int size = MatchingFlatSize(input1_shape, input2_shape, output_shape);
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

inline void Mul(const BinaryArithmeticOpParam &params, const Shape &input1_shape,
                const float *input1_data, const Shape &input2_shape, const float *input2_data,
                const Shape &output_shape, float *output_data)
{
  int i = 0;
  const int size = MatchingFlatSize(input1_shape, input2_shape, output_shape);
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

} // namespace optimized
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_OPTIMIZED_BINARYARITHMETICOPS_H__
