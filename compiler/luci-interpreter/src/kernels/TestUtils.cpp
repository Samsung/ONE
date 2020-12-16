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
    std::vector<uint8_t> data = extractTensorData<uint8_t>(tensor);
    return dequantize(data.data(), data.size(), tensor.scale(), tensor.zero_point());
  }
  else if (tensor.element_type() == DataType::S16)
  {
    // S16 quantization is symmetric, so zero point should be zero.
    for (auto zp : tensor.zero_points())
    {
      (void)zp;
      assert(zp == 0);
    }

    std::vector<int16_t> data = extractTensorData<int16_t>(tensor);
    if (tensor.scales().size() == 1)
    {
      return dequantize(data.data(), data.size(), tensor.scale(), 0);
    }

    // quantize_dimension breaks shape into two parts:
    // inner dimensions that contains continuous data with one quantization type
    // outer dimensions that contains other dimensions
    const Shape shape = tensor.shape();
    const int32_t quantized_dimension = tensor.quantized_dimension();
    assert(quantized_dimension < shape.num_dims());
    size_t outer_dims_size = 1;
    int32_t quant_dim_size = shape.dim(quantized_dimension);
    size_t inner_dims_size = 1;
    assert(quant_dim_size == tensor.scales().size());

    for (int i = 0; i < quantized_dimension; ++i)
      outer_dims_size *= shape.dim(i);
    for (int i = quantized_dimension + 1; i < shape.num_dims(); ++i)
      inner_dims_size *= shape.dim(i);

    assert(shape.num_elements() == outer_dims_size * quant_dim_size * inner_dims_size);

    std::vector<float> dequantized_data;
    dequantized_data.reserve(shape.num_elements());
    for (size_t outer_it = 0; outer_it < outer_dims_size; ++outer_it)
      for (int32_t channel = 0; channel < quant_dim_size; ++channel)
      {
        float scale = tensor.scales()[channel];
        size_t offset = inner_dims_size * (quant_dim_size * outer_it + channel);
        std::vector<float> part_dequantized_data =
          dequantize(data.data() + offset, inner_dims_size, scale, 0);
        dequantized_data.insert(dequantized_data.end(), part_dequantized_data.begin(),
                                part_dequantized_data.end());
      }
    return dequantized_data;
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
