/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef LUCI_INTERPRETER_KERNELS_TESTUTILS_H
#define LUCI_INTERPRETER_KERNELS_TESTUTILS_H

#include "luci_interpreter/core/Tensor.h"

#include <type_traits>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

namespace luci_interpreter
{
namespace kernels
{
namespace testing
{

template <DataType DT>
Tensor makeInputTensor(const Shape &shape, const std::vector<typename DataTypeImpl<DT>::Type> &data)
{
  Tensor tensor(DT, shape, {}, "");
  tensor.writeData(data.data(), data.size() * sizeof(typename DataTypeImpl<DT>::Type));
  return tensor;
}

Tensor makeOutputTensor(DataType element_type);
Tensor makeOutputTensor(DataType element_type, float scale, int32_t zero_point);

std::vector<int32_t> extractTensorShape(const Tensor &tensor);

// Returns the corresponding DataType given the type T.
template <typename T> constexpr DataType getElementType()
{
  if (std::is_same<T, float>::value)
    return DataType::FLOAT32;
  if (std::is_same<T, uint8_t>::value)
    return DataType::U8;
  if (std::is_same<T, int32_t>::value)
    return DataType::S32;
  if (std::is_same<T, int64_t>::value)
    return DataType::S64;
  return DataType::Unknown;
}

template <typename T> std::vector<T> extractTensorData(const Tensor &tensor)
{
  const auto *data_ptr = tensor.data<T>();
  return std::vector<T>(data_ptr, data_ptr + tensor.shape().num_elements());
}

std::vector<::testing::Matcher<float>> ArrayFloatNear(const std::vector<float> &values,
                                                      float max_abs_error = 1.0e-5f);

template <typename T>
inline std::vector<T> quantize(const std::vector<float> &data, float scale, int32_t zero_point)
{
  assert(!std::is_floating_point<T>::value);
  std::vector<T> q;
  for (const auto &f : data)
  {
    q.push_back(static_cast<T>(std::max<float>(
        std::numeric_limits<T>::min(),
        std::min<float>(std::numeric_limits<T>::max(), std::round(zero_point + (f / scale))))));
  }
  return q;
}

template <typename T>
inline std::vector<float> dequantize(const std::vector<T> &data, float scale, int32_t zero_point)
{
  assert(!std::is_floating_point<T>::value);
  std::vector<float> f;
  for (const T &q : data)
  {
    f.push_back(scale * (q - zero_point));
  }
  return f;
}

template <typename T> std::pair<float, int32_t> quantizationParams(float f_min, float f_max)
{
  if (std::is_floating_point<T>::value)
  {
    return {0.0, 0};
  }
  int32_t zero_point = 0;
  double scale = 0;
  const T qmin = std::numeric_limits<T>::min();
  const T qmax = std::numeric_limits<T>::max();
  const double qmin_double = qmin;
  const double qmax_double = qmax;
  // 0 should always be a representable value. Let's assume that the initial
  // min,max range contains 0.
  assert(f_max >= 0);
  assert(f_min <= 0);
  if (f_min == f_max)
  {
    // Special case where the min,max range is a point. Should be {0}.
    assert(f_max == 0);
    assert(f_min == 0);
    return {scale, zero_point};
  }

  // General case.
  //
  // First determine the scale.
  scale = (f_max - f_min) / (qmax_double - qmin_double);

  // Zero-point computation.
  // First the initial floating-point computation. The zero-point can be
  // determined from solving an affine equation for any known pair
  // (real value, corresponding quantized value).
  // We know two such pairs: (rmin, qmin) and (rmax, qmax).
  // The arithmetic error on the zero point computed from either pair
  // will be roughly machine_epsilon * (sum of absolute values of terms)
  // so we want to use the variant that adds the smaller terms.
  const double zero_point_from_min = qmin_double - f_min / scale;
  const double zero_point_from_max = qmax_double - f_max / scale;

  const double zero_point_from_min_error = std::abs(qmin_double) + std::abs(f_min / scale);

  const double zero_point_from_max_error = std::abs(qmax_double) + std::abs(f_max / scale);

  const double zero_point_double = zero_point_from_min_error < zero_point_from_max_error
                                       ? zero_point_from_min
                                       : zero_point_from_max;

  // Now we need to nudge the zero point to be an integer
  // (our zero points are integer, and this is motivated by the requirement
  // to be able to represent the real value "0" exactly as a quantized value,
  // which is required in multiple places, for example in Im2col with SAME
  //  padding).

  T nudged_zero_point = 0;
  if (zero_point_double < qmin_double)
  {
    nudged_zero_point = qmin;
  }
  else if (zero_point_double > qmax_double)
  {
    nudged_zero_point = qmax;
  }
  else
  {
    nudged_zero_point = static_cast<T>(std::round(zero_point_double));
  }

  // The zero point should always be in the range of quantized value,
  // // [qmin, qmax].
  assert(qmax >= nudged_zero_point);
  assert(qmin <= nudged_zero_point);
  zero_point = nudged_zero_point;
  // finally, return the values
  return {static_cast<float>(scale), zero_point};
}

inline float getTolerance(float min, float max, int quantize_steps)
{
  return ((max - min) / quantize_steps);
}

} // namespace testing
} // namespace kernels
} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_KERNELS_TESTUTILS_H
