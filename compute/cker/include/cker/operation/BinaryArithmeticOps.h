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
template <BinaryArithmeticOpType op_type, typename T, typename std::enable_if_t<!std::is_same<T, bool>::value, bool> = true>
const std::function<T(const T &, const T &)> GetBinaryArtithmeticFn()
{
  switch (op_type)
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
      if (std::is_floating_point<T>::value)
        return [](const T &a, const T &b) -> T { return a / b; };
      else
        return [](const T &a, const T &b) -> T {
          if (b == 0)
            throw std::runtime_error("Divide by zero");
          return a / b;
        };
    }
    case BinaryArithmeticOpType::POW:
    {
      return [](const T &a, const T &b) -> T { return std::pow(a, b); };
    }
    default:
    {
      assert(false);
      return nullptr;
    }
  }
}
template <BinaryArithmeticOpType op_type, typename T, typename std::enable_if_t<std::is_same<T, bool>::value, bool> = true>
const std::function<T(const bool &, const bool &)> GetBinaryArtithmeticFn()
{
  switch (op_type)
  {
    case BinaryArithmeticOpType::MUL:
    {
      return [](const bool &a, const bool &b) -> bool { return a && b; };
    }
    default:
    {
      throw std::runtime_error("GetBinaryArtithmeticFn: Unsupported OpType with Bool8");
    }
  }
}
} // namespace

// Consolidates dimensions in broadcast inputs, checks for five-fold pattern.
//
// For example, if sequence of dimensions of one input is
// ..., 1, 3, 1, 7, 9, 5,... and the other is ..., 2, 3, 1, 7, 1, 1, ...
// we can consolidate these as
// ..., 1, 3*7, 9*5, ... and 2, 3*7, 1.
//
// The category is updated in the less-frequent case of shapes that are
// not suited to a fivefold-loop broadcast.
//
// Falls back to generic pattern when it does not know how to process properly.
//
// Returns true iff there is some sort of broadcast, which includes five-fold
// patterns and falling back to generic broadcast.
inline bool ProcessBroadcastShapes(const Shape &shape0, const Shape &shape1,
                                   BinaryArithmeticOpParam *params)
{
  const int dims_count = std::max(shape0.DimensionsCount(), shape1.DimensionsCount());

  params->broadcast_category = BroadcastableOpCategory::kGenericBroadcast;
  Shape scalar_shape(dims_count, 1);

  auto extended_shape0 = Shape::ExtendedShape(dims_count, shape0);
  auto extended_shape1 = Shape::ExtendedShape(dims_count, shape1);

  // Check for "exact" match, implicitly accepting any scalar shapes.
  if (extended_shape0 == extended_shape1)
  {
    params->broadcast_category = BroadcastableOpCategory::kNonBroadcast;
    return false;
  }

  for (int i = dims_count - 1; i >= 0; --i)
  {
    if (extended_shape0.Dims(i) == extended_shape1.Dims(i))
    {
      continue;
    }
    else if (extended_shape0.Dims(i) == 1)
    {
      params->broadcast_category = BroadcastableOpCategory::kFirstInputBroadcastsFast;
      break;
    }
    else if (extended_shape1.Dims(i) == 1)
    {
      params->broadcast_category = BroadcastableOpCategory::kSecondInputBroadcastsFast;
      break;
    }
    else
    {
      // This case is erroneous: there is a dimension that does not match and
      // is not a broadcast from one shape to the other.
      params->broadcast_category = BroadcastableOpCategory::kGenericBroadcast;
      return true;
    }
  }

  if (params->broadcast_category != BroadcastableOpCategory::kFirstInputBroadcastsFast &&
      params->broadcast_category != BroadcastableOpCategory::kSecondInputBroadcastsFast)
  {
    return false;
  }

  // From this point it is assumed contractually that corresponding dimensions
  // in shape0 and shape1 are either (a) equal or (b) one or other equals 1.
  const bool swap_inputs =
    params->broadcast_category == BroadcastableOpCategory::kSecondInputBroadcastsFast;
  const Shape *shape_a = swap_inputs ? &extended_shape1 : &extended_shape0;
  const Shape *shape_b = swap_inputs ? &extended_shape0 : &extended_shape1;

  int i = dims_count - 1;
  params->broadcast_shape[0] = 1;
  params->broadcast_shape[1] = 1;
  params->broadcast_shape[2] = 1;
  params->broadcast_shape[3] = 1;
  params->broadcast_shape[4] = 1;
  // y_0 is greedy: include dims if both or neither equal 1: in other words,
  // test for equality rather than (shape_a->Dims(i) != 1).
  while (i >= 0 && shape_a->Dims(i) == shape_b->Dims(i))
  {
    params->broadcast_shape[4] *= shape_b->Dims(i);
    --i;
  }
  // Here either input_a or input_b has dim of 1 (if i >= 0).  If it is input_b
  // that has the unit dimension, the next two loops are not entered.
  while (i >= 0 && shape_a->Dims(i) == 1)
  {
    params->broadcast_shape[3] *= shape_b->Dims(i);
    --i;
  }
  while (i >= 0 && shape_a->Dims(i) == shape_b->Dims(i))
  {
    params->broadcast_shape[2] *= shape_a->Dims(i);
    --i;
  }
  // Here either input_a or input_b has dim of 1 (if i >= 0).
  while (i >= 0 && shape_b->Dims(i) == 1)
  {
    params->broadcast_shape[1] *= shape_a->Dims(i);
    --i;
  }
  while (i >= 0 && shape_a->Dims(i) == shape_b->Dims(i))
  {
    params->broadcast_shape[0] *= shape_b->Dims(i);
    --i;
  }

  // Rarer case is when the broadcast dimensions cannot be handled by a fivefold
  // loop.
  if (i >= 0)
  {
    params->broadcast_category = BroadcastableOpCategory::kGenericBroadcast;
  }
  return true;
}

template <BinaryArithmeticOpType op_type, typename T>
inline typename std::enable_if_t<!is_quant8<T>::value && std::is_same<T, bool>::value>
BinaryArithmeticOp(const BinaryArithmeticOpParam &params, const Shape &input1_shape,
                   const T *input1_data, const Shape &input2_shape, const T *input2_data,
                   const Shape &output_shape, T *output_data)
{
  reference::BinaryArithmeticOp(params, input1_shape, input1_data, input2_shape, input2_data,
                                output_shape, output_data, GetBinaryArtithmeticFn<op_type, bool>());
}

template <BinaryArithmeticOpType op_type, typename T>
inline typename std::enable_if_t<!is_quant8<T>::value && !std::is_same<T, bool>::value>
BinaryArithmeticOp(const BinaryArithmeticOpParam &params, const Shape &input1_shape,
                   const T *input1_data, const Shape &input2_shape, const T *input2_data,
                   const Shape &output_shape, T *output_data)
{
  reference::BinaryArithmeticOp(params, input1_shape, input1_data, input2_shape, input2_data,
                                output_shape, output_data, GetBinaryArtithmeticFn<op_type, T>());
}

template <BinaryArithmeticOpType op_type, typename T>
inline typename std::enable_if_t<is_quant8<T>::value>
BinaryArithmeticOp(const BinaryArithmeticOpParam &params, const Shape &input1_shape,
                   const T *input1_data, const Shape &input2_shape, const T *input2_data,
                   const Shape &output_shape, T *output_data)
{
  switch (op_type)
  {
    case nnfw::cker::BinaryArithmeticOpType::ADD:
    case nnfw::cker::BinaryArithmeticOpType::SUB:
      optimized::Add(params, input1_shape, input1_data, input2_shape, input2_data, output_shape,
                     output_data);
      break;
    case nnfw::cker::BinaryArithmeticOpType::MUL:
      optimized::Mul(params, input1_shape, input1_data, input2_shape, input2_data, output_shape,
                     output_data);
      break;
    case nnfw::cker::BinaryArithmeticOpType::DIV:
      throw std::runtime_error{"Quant8 Asymm NYI"};
    default:
      assert(false);
      break;
  }
}

template <BinaryArithmeticOpType op_type>
inline void BinaryArithmeticOp(const BinaryArithmeticOpParam &params, const Shape &input1_shape,
                               const float *input1_data, const Shape &input2_shape,
                               const float *input2_data, const Shape &output_shape,
                               float *output_data)
{
  // Supported type is only float now
  switch (op_type)
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
      optimized::Div(params, input1_shape, input1_data, input2_shape, input2_data, output_shape,
                     output_data);
      break;
    default:
      assert(false);
      break;
  }
}

template <BinaryArithmeticOpType op_type, typename T>
inline typename std::enable_if_t<!is_quant8<T>::value>
BroadcastBinaryArithmeticOp(BinaryArithmeticOpParam &params, const Shape &input1_shape,
                            const T *input1_data, const Shape &input2_shape, const T *input2_data,
                            const Shape &output_shape, T *output_data)
{
  reference::BroadcastBinaryArithmeticOpSlow(params, input1_shape, input1_data, input2_shape,
                                             input2_data, output_shape, output_data,
                                             GetBinaryArtithmeticFn<op_type, T>());
}

template <BinaryArithmeticOpType op_type, typename T>
inline typename std::enable_if_t<is_quant8<T>::value>
BroadcastBinaryArithmeticOp(BinaryArithmeticOpParam &params, const Shape &input1_shape,
                            const T *input1_data, const Shape &input2_shape, const T *input2_data,
                            const Shape &output_shape, T *output_data)
{
  switch (op_type)
  {
    case nnfw::cker::BinaryArithmeticOpType::ADD:
    case nnfw::cker::BinaryArithmeticOpType::SUB:
      optimized::BroadcastAddDispatch(params, input1_shape, input1_data, input2_shape, input2_data,
                                      output_shape, output_data);
      break;
    case nnfw::cker::BinaryArithmeticOpType::MUL:
      optimized::BroadcastMulDispatch(params, input1_shape, input1_data, input2_shape, input2_data,
                                      output_shape, output_data);
      break;
    case nnfw::cker::BinaryArithmeticOpType::DIV:
    case nnfw::cker::BinaryArithmeticOpType::POW:
      throw std::runtime_error{"Quant8 Asymm NYI"};
    default:
      assert(false);
      break;
  }
}

template <BinaryArithmeticOpType op_type>
inline void BroadcastBinaryArithmeticOp(BinaryArithmeticOpParam &params, const Shape &input1_shape,
                                        const float *input1_data, const Shape &input2_shape,
                                        const float *input2_data, const Shape &output_shape,
                                        float *output_data)
{
  // Supported type is only float now
  switch (op_type)
  {
    case nnfw::cker::BinaryArithmeticOpType::ADD:
      optimized::BroadcastAddDispatch(params, input1_shape, input1_data, input2_shape, input2_data,
                                      output_shape, output_data);
      break;
    case nnfw::cker::BinaryArithmeticOpType::MUL:
      optimized::BroadcastMulDispatch(params, input1_shape, input1_data, input2_shape, input2_data,
                                      output_shape, output_data);
      break;
    case nnfw::cker::BinaryArithmeticOpType::SUB:
      optimized::BroadcastSubDispatch(params, input1_shape, input1_data, input2_shape, input2_data,
                                      output_shape, output_data);
      break;
    case nnfw::cker::BinaryArithmeticOpType::DIV:
      optimized::BroadcastDivDispatch(params, input1_shape, input1_data, input2_shape, input2_data,
                                      output_shape, output_data);
      break;
    case nnfw::cker::BinaryArithmeticOpType::POW:
      reference::BroadcastBinaryArithmeticOpSlow<float>(
        params, input1_shape, input1_data, input2_shape, input2_data, output_shape, output_data,
        GetBinaryArtithmeticFn<op_type, float>());
      break;
    default:
      assert(false);
      break;
  }
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_BINARY_ARITHMETIC_OPS_H__
