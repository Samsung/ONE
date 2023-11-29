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

#ifndef __NNFW_CKER_REFERENCE_BINARYARITHMETICOPS_H__
#define __NNFW_CKER_REFERENCE_BINARYARITHMETICOPS_H__

#include "cker/Shape.h"
#include "cker/Types.h"
#include "cker/Utils.h"

#include <cmath>

namespace nnfw
{
namespace cker
{
namespace reference
{

template <typename T>
inline void BinaryArithmeticOp(const BinaryArithmeticOpParam &params, const Shape &input1_shape,
                               const T *input1_data, const Shape &input2_shape,
                               const T *input2_data, const Shape &output_shape, T *output_data,
                               const std::function<T(const T &, const T &)> &fn)
{
  const int32_t flat_size = MatchingElementsSize(input1_shape, input2_shape, output_shape);
  for (int i = 0; i < flat_size; ++i)
  {
    output_data[i] = ActivationFunctionWithMinMax(fn(input1_data[i], input2_data[i]),
                                                  params.quantized_activation_min,
                                                  params.quantized_activation_max);
  }
}

template <>
inline void BinaryArithmeticOp(const BinaryArithmeticOpParam &params, const Shape &input1_shape,
                               const float *input1_data, const Shape &input2_shape,
                               const float *input2_data, const Shape &output_shape,
                               float *output_data,
                               const std::function<float(const float &, const float &)> &fn)
{
  const int size = MatchingElementsSize(input1_shape, input2_shape, output_shape);
  for (int i = 0; i < size; i++)
  {
    output_data[i] = ActivationFunctionWithMinMax(
      fn(input1_data[i], input2_data[i]), params.float_activation_min, params.float_activation_max);
  }
}

template <>
inline void BinaryArithmeticOp(const BinaryArithmeticOpParam &, const Shape &input1_shape,
                               const bool *input1_data, const Shape &input2_shape,
                               const bool *input2_data, const Shape &output_shape,
                               bool *output_data,
                               const std::function<bool(const bool &, const bool &)> &fn)
{
  const int size = MatchingElementsSize(input1_shape, input2_shape, output_shape);
  for (int i = 0; i < size; i++)
  {
    output_data[i] = fn(input1_data[i], input2_data[i]);
  }
}

template <typename T>
inline typename std::enable_if_t<is_quant8<T>::value> BroadcastBinaryArithmeticOpSlow(
  const BinaryArithmeticOpParam &params, const Shape &input1_shape, const T *input1_data,
  const Shape &input2_shape, const T *input2_data, const Shape &output_shape, T *output_data,
  const std::function<T(const BinaryArithmeticOpParam &params, const T &, const T &)> &fn)
{
  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_shape, input2_shape, &desc1, &desc2);
  const Shape extended_output_shape = Shape::ExtendedShape(4, output_shape);

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
          output_data[Offset(extended_output_shape, b, y, x, c)] = ActivationFunctionWithMinMax<T>(
            fn(params, input1_data[SubscriptToIndex(desc1, b, y, x, c)],
               input2_data[SubscriptToIndex(desc2, b, y, x, c)]),
            params.quantized_activation_min, params.quantized_activation_max);
        }
      }
    }
  }
}
template <typename T>
inline void BroadcastBinaryArithmeticOpSlow(const BinaryArithmeticOpParam &params,
                                            const Shape &input1_shape, const T *input1_data,
                                            const Shape &input2_shape, const T *input2_data,
                                            const Shape &output_shape, T *output_data,
                                            const std::function<T(const T &, const T &)> &fn)
{
  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_shape, input2_shape, &desc1, &desc2);
  const Shape extended_output_shape = Shape::ExtendedShape(4, output_shape);

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
          output_data[Offset(extended_output_shape, b, y, x, c)] = ActivationFunctionWithMinMax<T>(
            fn(input1_data[SubscriptToIndex(desc1, b, y, x, c)],
               input2_data[SubscriptToIndex(desc2, b, y, x, c)]),
            params.quantized_activation_min, params.quantized_activation_max);
        }
      }
    }
  }
}

template <>
inline void BroadcastBinaryArithmeticOpSlow(
  const BinaryArithmeticOpParam &params, const Shape &input1_shape, const float *input1_data,
  const Shape &input2_shape, const float *input2_data, const Shape &output_shape,
  float *output_data, const std::function<float(const float &, const float &)> &fn)
{
  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_shape, input2_shape, &desc1, &desc2);
  const Shape extended_output_shape = Shape::ExtendedShape(4, output_shape);

  for (int b = 0; b < extended_output_shape.Dims(0); ++b)
  {
    for (int y = 0; y < extended_output_shape.Dims(1); ++y)
    {
      for (int x = 0; x < extended_output_shape.Dims(2); ++x)
      {
        for (int c = 0; c < extended_output_shape.Dims(3); ++c)
        {
          output_data[Offset(extended_output_shape, b, y, x, c)] =
            ActivationFunctionWithMinMax(fn(input1_data[SubscriptToIndex(desc1, b, y, x, c)],
                                            input2_data[SubscriptToIndex(desc2, b, y, x, c)]),
                                         params.float_activation_min, params.float_activation_max);
        }
      }
    }
  }
}

template <>
inline void BroadcastBinaryArithmeticOpSlow(
  const BinaryArithmeticOpParam &, const Shape &input1_shape, const bool *input1_data,
  const Shape &input2_shape, const bool *input2_data, const Shape &output_shape, bool *output_data,
  const std::function<bool(const bool &, const bool &)> &fn)
{
  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_shape, input2_shape, &desc1, &desc2);
  const Shape extended_output_shape = Shape::ExtendedShape(4, output_shape);

  for (int b = 0; b < extended_output_shape.Dims(0); ++b)
  {
    for (int y = 0; y < extended_output_shape.Dims(1); ++y)
    {
      for (int x = 0; x < extended_output_shape.Dims(2); ++x)
      {
        for (int c = 0; c < extended_output_shape.Dims(3); ++c)
        {
          output_data[Offset(extended_output_shape, b, y, x, c)] =
            fn(input1_data[SubscriptToIndex(desc1, b, y, x, c)],
               input2_data[SubscriptToIndex(desc2, b, y, x, c)]);
        }
      }
    }
  }
}

} // namespace reference
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_REFERENCE_BINARYARITHMETICOPS_H__
