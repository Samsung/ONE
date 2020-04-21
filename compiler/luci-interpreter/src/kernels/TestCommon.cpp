/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

#include "kernels/TestCommon.h"

namespace luci_interpreter
{
namespace testing
{

std::unique_ptr<Tensor> getEmptyTensor(DataType DT, const Shape &shape, AffineQuantization quant)
{
  return std::make_unique<Tensor>(DT, shape, std::move(quant), "test_tensor");
}

std::vector<float> extractDequantizedData(const Tensor *tensor)
{
  const auto *data = tensor->data<uint8_t>();
  std::vector<float> res;
  const int32_t num_elements = tensor->shape().num_elements();

  const float scale = tensor->scale();
  const int32_t zero_point = tensor->zero_point();

  for (int32_t i = 0; i < num_elements; ++i)
  {
    res.push_back(scale * static_cast<float>(data[i] - zero_point));
  }
  return res;
}

std::vector<::testing::Matcher<float>> ArrayFloatNear(const std::vector<float> &values,
                                                      float max_abs_error)
{
  std::vector<::testing::Matcher<float>> matchers;
  matchers.reserve(values.size());
  for (const float v : values)
  {
    matchers.emplace_back(::testing::FloatNear(v, max_abs_error));
  }
  return matchers;
}

} // namespace testing
} // namespace luci_interpreter
