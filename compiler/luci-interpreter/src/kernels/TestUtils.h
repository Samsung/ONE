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
#include "luci_interpreter/MemoryManager.h"

#include <type_traits>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

namespace luci_interpreter
{
namespace kernels
{
namespace testing
{

template <typename T>
std::vector<T> quantize(const float *data, size_t num_elements, float scale, int32_t zero_point);

template <DataType DT>
Tensor makeInputTensor(const Shape &shape, const std::vector<typename DataTypeImpl<DT>::Type> &data,
                       IMemoryManager *memory_manager)
{
  Tensor tensor(DT, shape, {}, "");
  memory_manager->allocate_memory(tensor);
  tensor.writeData(data.data(), data.size() * sizeof(typename DataTypeImpl<DT>::Type));
  return tensor;
}

template <DataType DT>
Tensor makeInputTensor(const Shape &shape, const std::vector<typename DataTypeImpl<DT>::Type> &data)
{
  Tensor tensor(DT, shape, {}, "");
  tensor.writeData(data.data(), data.size() * sizeof(typename DataTypeImpl<DT>::Type));
  return tensor;
}

/**
 * @brief Create layer-wise quantized tensor
 * @tparam DT base integer data type, for example DataType::U8, DataType::S16, DataType::S64
 * @param shape desired tensor shape
 * @param scale scale of quantized number
 * @param zero_point zero point of quantized number, should be 0 for signed datatypes
 * @param data floating point data for quantization
 * @return created tensor
 */
template <DataType DT>
Tensor makeInputTensor(const Shape &shape, float scale, int32_t zero_point,
                       const std::vector<float> &data, IMemoryManager *memory_manager)
{
  using NativeT = typename DataTypeImpl<DT>::Type;
  Tensor tensor(DT, shape, {{scale}, {zero_point}}, "");
  std::vector<NativeT> quantized_data =
    quantize<NativeT>(data.data(), data.size(), scale, zero_point);
  memory_manager->allocate_memory(tensor);
  tensor.writeData(quantized_data.data(), quantized_data.size() * sizeof(NativeT));
  return tensor;
}

template <DataType DT>
Tensor makeInputTensor(const Shape &shape, float scale, int32_t zero_point,
                       const std::vector<float> &data)
{
  using NativeT = typename DataTypeImpl<DT>::Type;
  Tensor tensor(DT, shape, {{scale}, {zero_point}}, "");
  std::vector<NativeT> quantized_data =
    quantize<NativeT>(data.data(), data.size(), scale, zero_point);
  tensor.writeData(quantized_data.data(), quantized_data.size() * sizeof(NativeT));
  return tensor;
}

/**
 * @brief Create channel-wise quantized tensor
 * @tparam DT base integer data type, for example DataType::U8, DataType::S16, DataType::S64
 * @param shape desired tensor shape
 * @param scales scales of quantized number
 * @param zero_points zero points of quantized number, should be 0 for signed datatypes
 * @param quantize_dimension dimension to apply quantization along. Usually channels/output channels
 * @param data floating point data for quantization
 * @return created tensor
 */
template <DataType DT>
Tensor makeInputTensor(const Shape &shape, const std::vector<float> &scales,
                       const std::vector<int32_t> &zero_points, int quantized_dimension,
                       const std::vector<float> &data, IMemoryManager *memory_manager = nullptr)
{
  using NativeT = typename DataTypeImpl<DT>::Type;
  assert(quantized_dimension < shape.num_dims());
  Tensor tensor(DT, shape, {scales, zero_points, quantized_dimension}, "");

  // quantize_dimension breaks shape into two parts:
  // inner dimensions that contains continuous data with one quantization type
  // outer dimensions that contains other dimensions
  size_t outer_dims_size = 1;
  int32_t quant_dim_size = shape.dim(quantized_dimension);
  size_t inner_dims_size = 1;
  assert(quant_dim_size == scales.size());
  assert(quant_dim_size == zero_points.size());

  for (int i = 0; i < quantized_dimension; ++i)
    outer_dims_size *= shape.dim(i);
  for (int i = quantized_dimension + 1; i < shape.num_dims(); ++i)
    inner_dims_size *= shape.dim(i);

  assert(shape.num_elements() == outer_dims_size * quant_dim_size * inner_dims_size);

  std::vector<NativeT> quantized_data;
  quantized_data.reserve(shape.num_elements());
  for (size_t outer_it = 0; outer_it < outer_dims_size; ++outer_it)
    for (int32_t channel = 0; channel < quant_dim_size; ++channel)
    {
      int32_t zero_point = zero_points[channel];
      float scale = scales[channel];
      size_t offset = inner_dims_size * (quant_dim_size * outer_it + channel);
      std::vector<NativeT> part_quantized_data =
        quantize<NativeT>(data.data() + offset, inner_dims_size, scale, zero_point);
      quantized_data.insert(quantized_data.end(), part_quantized_data.begin(),
                            part_quantized_data.end());
    }
  assert(quantized_data.size() == shape.num_elements());
  memory_manager->allocate_memory(tensor);
  tensor.writeData(quantized_data.data(), quantized_data.size() * sizeof(NativeT));
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
  if (std::is_same<T, double>::value)
    return DataType::FLOAT64;
  if (std::is_same<T, uint8_t>::value)
    return DataType::U8;
  if (std::is_same<T, uint16_t>::value)
    return DataType::U16;
  if (std::is_same<T, uint32_t>::value)
    return DataType::U32;
  if (std::is_same<T, uint64_t>::value)
    return DataType::U64;
  if (std::is_same<T, int8_t>::value)
    return DataType::S8;
  if (std::is_same<T, int16_t>::value)
    return DataType::S16;
  if (std::is_same<T, int32_t>::value)
    return DataType::S32;
  if (std::is_same<T, int64_t>::value)
    return DataType::S64;
  if (std::is_same<T, bool>::value)
    return DataType::BOOL;
  return DataType::Unknown;
}

template <typename T> std::vector<T> extractTensorData(const Tensor &tensor)
{
  const auto *data_ptr = tensor.data<T>();
  return std::vector<T>(data_ptr, data_ptr + tensor.shape().num_elements());
}

std::vector<float> dequantizeTensorData(const Tensor &tensor);

// Array version of `::testing::FloatNear` matcher.
::testing::Matcher<std::vector<float>> FloatArrayNear(const std::vector<float> &values,
                                                      float max_abs_error = 1.0e-5f);

template <typename T>
std::vector<T> quantize(const float *data, size_t num_elements, float scale, int32_t zero_point)
{
  static_assert(std::is_integral<T>::value, "Integral type expected.");

  float q_min{}, q_max{};
  if (std::is_signed<T>::value)
  {
    // For now, assume that signed type implies signed symmetric quantization.
    assert(zero_point == 0);
    q_min = -std::numeric_limits<T>::max();
    q_max = std::numeric_limits<T>::max();
  }
  else
  {
    q_min = 0;
    q_max = std::numeric_limits<T>::max();
  }

  std::vector<T> q;
  for (size_t i = 0; i < num_elements; ++i)
  {
    const auto &f = data[i];
    q.push_back(static_cast<T>(
      std::max<float>(q_min, std::min<float>(q_max, std::round(zero_point + (f / scale))))));
  }
  return q;
}

template <typename T>
std::vector<float> dequantize(const T *data, size_t num_elements, float scale, int32_t zero_point)
{
  static_assert(std::is_integral<T>::value, "Integral type expected.");
  std::vector<float> f;
  for (size_t i = 0; i < num_elements; ++i)
  {
    const T &q = data[i];
    f.push_back(scale * (q - zero_point));
  }
  return f;
}

// NOTE Returns scale and zero point for _asymmetric_ range (both signed and unsigned).
template <typename T> std::pair<float, int32_t> quantizationParams(float f_min, float f_max)
{
  static_assert(std::is_integral<T>::value, "Integral type expected.");
  int32_t zero_point = 0;
  float scale = 0;
  const T qmin = std::numeric_limits<T>::lowest();
  const T qmax = std::numeric_limits<T>::max();
  const float qmin_double = qmin;
  const float qmax_double = qmax;
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
  const float zero_point_from_min = qmin_double - f_min / scale;
  const float zero_point_from_max = qmax_double - f_max / scale;

  const float zero_point_from_min_error = std::abs(qmin_double) + std::abs(f_min / scale);

  const float zero_point_from_max_error = std::abs(qmax_double) + std::abs(f_max / scale);

  const float zero_point_double = zero_point_from_min_error < zero_point_from_max_error
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
  return {scale, zero_point};
}

inline float getTolerance(float min, float max, int quantize_steps)
{
  return ((max - min) / quantize_steps);
}

} // namespace testing
} // namespace kernels
} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_KERNELS_TESTUTILS_H
