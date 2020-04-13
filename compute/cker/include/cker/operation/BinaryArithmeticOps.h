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

#ifndef __NNFW_CKER_BINARY_ARITHMETIC_OPS_H__
#define __NNFW_CKER_BINARY_ARITHMETIC_OPS_H__

#include <functional>
#include "cker/operation/optimized/BinaryArithmeticOps.h"
#include "cker/operation/reference/BinaryArithmeticOps.h"
#include "cker/Shape.h"
#include "cker/Types.h"
#include "cker/Utils.h"

namespace nnfw
{
namespace cker
{

namespace
{
template <typename T>
const std::function<T(const T &, const T &)> GetBinaryArtithmeticFn(BinaryArithmeticOpType type)
{
  switch (type)
  {
    case BinaryArithmeticOpType::ADD:
    {
      return [](const T &a, const T &b) -> T { return a + b; };
    }
    case BinaryArithmeticOpType::MUL:
    {
      return [](const T &a, const T &b) -> T { return a * b; };
    }
    case BinaryArithmeticOpType::SUB:
    {
      return [](const T &a, const T &b) -> T { return a - b; };
    }
    case BinaryArithmeticOpType::DIV:
    {
      return [](const T &a, const T &b) -> T {
        if (b == 0)
        {
          throw std::runtime_error("Divide by zero");
        }
        return a / b;
      };
    }
    default:
    {
      assert(false);
      return nullptr;
    }
  }
}
} // namespace

template <typename T>
inline void BinaryArithmeticOp(const BinaryArithmeticOpParam &params, const Shape &input1_shape,
                               const T *input1_data, const Shape &input2_shape,
                               const T *input2_data, const Shape &output_shape, T *output_data)
{
  reference::BinaryArithmeticOp(params, input1_shape, input1_data, input2_shape, input2_data,
                                output_shape, output_data, GetBinaryArtithmeticFn<T>(params.type));
}

template <>
inline void BinaryArithmeticOp(const BinaryArithmeticOpParam &params, const Shape &input1_shape,
                               const float *input1_data, const Shape &input2_shape,
                               const float *input2_data, const Shape &output_shape,
                               float *output_data)
{
  // Supported type is only float now
  switch (params.type)
  {
    case nnfw::cker::BinaryArithmeticOpType::ADD:
      optimized::Add(params, input1_shape, input1_data, input2_shape, input2_data, output_shape,
                     output_data);
      break;
    case nnfw::cker::BinaryArithmeticOpType::MUL:
      optimized::Mul(params, input1_shape, input1_data, input2_shape, input2_data, output_shape,
                     output_data);
      break;
    case nnfw::cker::BinaryArithmeticOpType::SUB:
      optimized::Sub(params, input1_shape, input1_data, input2_shape, input2_data, output_shape,
                     output_data);
      break;
    case nnfw::cker::BinaryArithmeticOpType::DIV:
      reference::BinaryArithmeticOp(params, input1_shape, input1_data, input2_shape, input2_data,
                                    output_shape, output_data,
                                    GetBinaryArtithmeticFn<float>(params.type));
      break;
    default:
      assert(false);
      break;
  }
}

template <typename T>
inline void BroadcastBinaryArithmeticOpSlow(const BinaryArithmeticOpParam &params,
                                            const Shape &input1_shape, const T *input1_data,
                                            const Shape &input2_shape, const T *input2_data,
                                            const Shape &output_shape, T *output_data)
{
  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_shape, input2_shape, &desc1, &desc2);
  const Shape extended_output_shape = Shape::ExtendedShape(4, output_shape);

  const auto fn = GetBinaryArtithmeticFn<T>(params.type);

  // Comment from tensorflow lite:
  //
  // In Tensorflow, the dimensions are canonically named (batch_number, row,
  // col, channel), with extents (batches, height, width, depth), with the
  // trailing dimension changing most rapidly (channels has the smallest stride,
  // typically 1 element).
  //
  // In generated C code, we store arrays with the dimensions reversed. The
  // first dimension has smallest stride.
  //
  // We name our variables by their Tensorflow convention, but generate C code
  // nesting loops such that the innermost loop has the smallest stride for the
  // best cache behavior.
  for (int b = 0; b < extended_output_shape.Dims(0); ++b)
  {
    for (int y = 0; y < extended_output_shape.Dims(1); ++y)
    {
      for (int x = 0; x < extended_output_shape.Dims(2); ++x)
      {
        for (int c = 0; c < extended_output_shape.Dims(3); ++c)
        {
          output_data[Offset(extended_output_shape, b, y, x, c)] = ActivationFunctionWithMinMax(
              fn(input1_data[SubscriptToIndex(desc1, b, y, x, c)],
                 input2_data[SubscriptToIndex(desc2, b, y, x, c)]),
              params.quantized_activation_min, params.quantized_activation_max);
        }
      }
    }
  }
}

template <>
inline void BroadcastBinaryArithmeticOpSlow(const BinaryArithmeticOpParam &params,
                                            const Shape &input1_shape, const float *input1_data,
                                            const Shape &input2_shape, const float *input2_data,
                                            const Shape &output_shape, float *output_data)
{
  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_shape, input2_shape, &desc1, &desc2);
  const Shape extended_output_shape = Shape::ExtendedShape(4, output_shape);

  const auto fn = GetBinaryArtithmeticFn<float>(params.type);

  for (int b = 0; b < extended_output_shape.Dims(0); ++b)
  {
    for (int y = 0; y < extended_output_shape.Dims(1); ++y)
    {
      for (int x = 0; x < extended_output_shape.Dims(2); ++x)
      {
        for (int c = 0; c < extended_output_shape.Dims(3); ++c)
        {
          output_data[Offset(extended_output_shape, b, y, x, c)] = ActivationFunctionWithMinMax(
              fn(input1_data[SubscriptToIndex(desc1, b, y, x, c)],
                 input2_data[SubscriptToIndex(desc2, b, y, x, c)]),
              params.float_activation_min, params.float_activation_max);
        }
      }
    }
  }
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_BINARY_ARITHMETIC_OPS_H__
