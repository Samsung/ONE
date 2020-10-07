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

#include "kernels/TestUtils.h"

#include <stdexcept>

namespace luci_interpreter
{
namespace kernels
{
namespace testing
{

using ::testing::FloatNear;
using ::testing::Matcher;

Tensor makeOutputTensor(DataType element_type) { return Tensor(element_type, {}, {}, ""); }

Tensor makeOutputTensor(DataType element_type, float scale, int32_t zero_point)
{
  return Tensor(element_type, {}, {{scale}, {zero_point}}, "");
}

std::vector<float> dequantizeTensorData(const Tensor &tensor)
{
  if (tensor.element_type() == DataType::U8)
  {
    return dequantize(extractTensorData<uint8_t>(tensor), tensor.scale(), tensor.zero_point());
  }
  else if (tensor.element_type() == DataType::S16)
  {
    // S16 quantization is symmetric, so zero point should be zero.
    assert(tensor.zero_point() == 0);
    return dequantize(extractTensorData<int16_t>(tensor), tensor.scale(), 0);
  }
  else
  {
    throw std::runtime_error("Unsupported type.");
  }
}

Matcher<std::vector<float>> FloatArrayNear(const std::vector<float> &values, float max_abs_error)
{
  std::vector<Matcher<float>> matchers;
  matchers.reserve(values.size());
  for (const float v : values)
  {
    matchers.emplace_back(FloatNear(v, max_abs_error));
  }
  return ElementsAreArray(matchers);
}

std::vector<int32_t> extractTensorShape(const Tensor &tensor)
{
  std::vector<int32_t> result;
  int dims = tensor.shape().num_dims();
  for (int i = 0; i < dims; i++)
  {
    result.push_back(tensor.shape().dim(i));
  }
  return result;
}

} // namespace testing
} // namespace kernels
} // namespace luci_interpreter
