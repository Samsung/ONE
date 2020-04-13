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

#include "Quantization.h"
#include "mir/Tensor.h"
#include "mir/ShapeRange.h"

#include <cmath>
#include <limits>

namespace mir_interpreter
{
using namespace mir;

void Dequantize(const TensorVariant &input, TensorVariant &output)
{
  const TensorType &input_type = input.getType();
  assert(input_type.isQuantized());
  assert(input_type.getElementType() == DataType::UINT8);

  const float scale = input_type.getQuantization().getScale();
  const int32_t zero_point = input_type.getQuantization().getZeroPoint();

  Tensor<uint8_t> input_accessor(input);
  Tensor<float> res_accessor(output);

  for (const auto &index : ShapeRange(output.getShape()))
  {
    const int32_t value = input_accessor.at(index);
    res_accessor.at(index) = scale * static_cast<float>(value - zero_point);
  }
}

void Quantize(const TensorVariant &input, TensorVariant &output)
{
  const TensorType &output_type = output.getType();
  assert(output_type.isQuantized());
  assert(input.getElementType() == DataType::FLOAT32);

  const float scale = output_type.getQuantization().getScale();
  const int32_t zero_point = output_type.getQuantization().getZeroPoint();

  const int32_t min_val = std::numeric_limits<uint8_t>::min();
  const int32_t max_val = std::numeric_limits<uint8_t>::max();

  Tensor<float> input_accessor(input);
  Tensor<uint8_t> res_accessor(output);

  for (const auto &index : ShapeRange(output.getShape()))
  {
    const float value = input_accessor.at(index);
    int32_t unclamped = static_cast<int32_t>(std::round(value / scale)) + zero_point;
    int32_t clamped = std::min(std::max(unclamped, min_val), max_val);
    res_accessor.at(index) = static_cast<uint8_t>(clamped);
  }
}

} // namespace mir_interpreter
