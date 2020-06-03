/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

// Returns the corresponding DataType given the type T.
template <typename T> DataType getElementType()
{
  if (std::is_same<T, float>::value)
    return DataType::FLOAT32;
  if (std::is_same<T, uint8_t>::value)
    return DataType::U8;
  return DataType::Unknown;
}

template <typename T> std::vector<T> extractTensorData(const Tensor &tensor)
{
  const auto *data_ptr = tensor.data<T>();
  return std::vector<T>(data_ptr, data_ptr + tensor.shape().num_elements());
}

std::vector<::testing::Matcher<float>> ArrayFloatNear(const std::vector<float> &values,
                                                      float max_abs_error = 1.0e-5f);

} // namespace testing
} // namespace kernels
} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_KERNELS_TESTUTILS_H
