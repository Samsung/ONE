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

#include "Add.h"
#include "Common.h"

#include "QuantizationHelpers.h"
#include "mir/Tensor.h"
#include "mir/ShapeRange.h"

#include <cmath>

namespace mir_interpreter
{

using namespace mir;

template <typename T> struct AddImpl
{
  static void run(const TensorVariant &lhs, const TensorVariant &rhs, TensorVariant &res);
};

template <typename T>
void AddImpl<T>::run(const TensorVariant &lhs, const TensorVariant &rhs, TensorVariant &res)
{
  TensorVariant broadcasted_lhs(lhs, res.getShape());
  TensorVariant broadcasted_rhs(rhs, res.getShape());
  Tensor<T> lhs_accessor(broadcasted_lhs);
  Tensor<T> rhs_accessor(broadcasted_rhs);
  Tensor<T> res_accessor(res);

  for (const auto &index : ShapeRange(res.getShape()))
  {
    res_accessor.at(index) = lhs_accessor.at(index) + rhs_accessor.at(index);
  }
}

template <> struct AddImpl<uint8_t>
{
  static void run(const TensorVariant &lhs, const TensorVariant &rhs, TensorVariant &res);
};

void AddImpl<uint8_t>::run(const TensorVariant &lhs, const TensorVariant &rhs, TensorVariant &res)
{
  const auto &lhs_type = lhs.getType();
  const auto &rhs_type = rhs.getType();
  const auto &res_type = res.getType();

  assert(lhs_type.isQuantized());
  assert(rhs_type.isQuantized());
  assert(res_type.isQuantized());

  int32_t lhs_offset = -lhs_type.getQuantization().getZeroPoint();
  int32_t rhs_offset = -rhs_type.getQuantization().getZeroPoint();
  int32_t output_offset = res_type.getQuantization().getZeroPoint();

  double lhs_scale = lhs_type.getQuantization().getScale();
  double rhs_scale = rhs_type.getQuantization().getScale();
  double output_scale = res_type.getQuantization().getScale();

  int left_shift = 20;
  const double twice_max_input_scale = 2 * std::max(lhs_scale, rhs_scale);
  const double real_lhs_multiplier = lhs_scale / twice_max_input_scale;
  const double real_rhs_multiplier = rhs_scale / twice_max_input_scale;
  const double real_output_multiplier = twice_max_input_scale / ((1 << left_shift) * output_scale);

  int32_t lhs_multiplier = 0;
  int32_t rhs_multiplier = 0;
  int32_t output_multiplier = 0;
  int lhs_shift = 0;
  int rhs_shift = 0;
  int output_shift = 0;

  QuantizeMultiplierSmallerThanOneExp(real_lhs_multiplier, &lhs_multiplier, &lhs_shift);
  QuantizeMultiplierSmallerThanOneExp(real_rhs_multiplier, &rhs_multiplier, &rhs_shift);
  QuantizeMultiplierSmallerThanOneExp(real_output_multiplier, &output_multiplier, &output_shift);

  TensorVariant broadcasted_lhs(lhs, res_type.getShape());
  TensorVariant broadcasted_rhs(rhs, res_type.getShape());

  Tensor<uint8_t> lhs_accessor(broadcasted_lhs);
  Tensor<uint8_t> rhs_accessor(broadcasted_rhs);
  Tensor<uint8_t> res_accessor(res);

  int32_t output_min = std::numeric_limits<uint8_t>::min();
  int32_t output_max = std::numeric_limits<uint8_t>::max();

  for (const auto &index : ShapeRange(res_type.getShape()))
  {
    const int32_t lhs_val = lhs_accessor.at(index) + lhs_offset;
    const int32_t rhs_val = rhs_accessor.at(index) + rhs_offset;
    const int32_t shifted_lhs_val = lhs_val * (1 << left_shift);
    const int32_t shifted_rhs_val = rhs_val * (1 << left_shift);
    const int32_t scaled_lhs_val =
      MultiplyByQuantizedMultiplierSmallerThanOneExp(shifted_lhs_val, lhs_multiplier, lhs_shift);
    const int32_t scaled_rhs_val =
      MultiplyByQuantizedMultiplierSmallerThanOneExp(shifted_rhs_val, rhs_multiplier, rhs_shift);
    const int32_t raw_sum = scaled_lhs_val + scaled_rhs_val;
    const int32_t raw_output =
      MultiplyByQuantizedMultiplierSmallerThanOneExp(raw_sum, output_multiplier, output_shift) +
      output_offset;
    const int32_t clamped_output = std::min(output_max, std::max(output_min, raw_output));
    res_accessor.at(index) = static_cast<uint8_t>(clamped_output);
  }
}

void Add(const TensorVariant &lhs, const TensorVariant &rhs, TensorVariant &res)
{
  if (lhs.getElementType() != rhs.getElementType())
  {
    throw std::runtime_error{"Add with different input types is unsupported"};
  }
  dispatch<AddImpl>(res.getElementType(), lhs, rhs, res);
}

} // namespace mir_interpreter
