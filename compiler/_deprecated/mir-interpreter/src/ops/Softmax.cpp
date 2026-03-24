/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Softmax.h"
#include "Common.h"
#include "QuantizationHelpers.h"

#include <mir/ShapeRange.h>
#include <mir/Tensor.h>

#include <cmath>

namespace mir_interpreter
{

static inline void PopulateSoftmaxLookupTable(float *table, float input_scale, float beta)
{
  const float scale = -input_scale * beta;
  const int32_t max_uint8 = std::numeric_limits<uint8_t>::max();
  for (int32_t val = 0; val <= max_uint8; ++val)
    table[max_uint8 - val] = expf(scale * val);
}

template <typename T> struct SoftmaxImpl
{
  static void run(const mir::TensorVariant &arg, int axis, mir::TensorVariant &result);
};

template <typename T>
void SoftmaxImpl<T>::run(const mir::TensorVariant &arg, int axis, mir::TensorVariant &result)
{
  mir::Tensor<T> arg_accessor(arg);
  mir::Tensor<T> res_accessor(result);

  mir::Shape expsum_shape = arg.getShape();
  expsum_shape.dim(axis) = 1;
  mir::TensorType expsum_type(arg.getElementType(), expsum_shape);
  mir::TensorVariant expsum(expsum_type);
  mir::Tensor<T> expsum_accessor(expsum);

  for (const auto &expsum_index : mir::ShapeRange(expsum_shape))
  {
    T sum = 0;
    mir::Index arg_index = expsum_index;
    std::int32_t axis_size = arg.getShape().dim(axis);
    for (std::int32_t i = 0; i < axis_size; ++i)
    {
      arg_index.at(axis) = i;
      sum += std::exp(arg_accessor.at(arg_index));
    }
    expsum_accessor.at(expsum_index) = sum;
  }

  for (const auto &res_index : mir::ShapeRange(result.getShape()))
  {
    mir::Index expsum_index = res_index;
    expsum_index.at(axis) = 0;
    res_accessor.at(res_index) =
      std::exp(arg_accessor.at(res_index)) / expsum_accessor.at(expsum_index);
  }
}

template <> struct SoftmaxImpl<uint8_t>
{
  static void run(const mir::TensorVariant &input, int axis, mir::TensorVariant &output);
};

void SoftmaxImpl<uint8_t>::run(const mir::TensorVariant &input, int axis,
                               mir::TensorVariant &output)
{
  const auto &input_type = input.getType();
  const auto &output_type = output.getType();

  assert(input_type.isQuantized());
  assert(output_type.isQuantized());

  const auto input_shape = input_type.getShape();

  assert(input_type.getElementType() == mir::DataType::UINT8);
  assert(axis == input_shape.rank() - 1); // supported only last dim axis
  (void)axis;

  double input_scale = input_type.getQuantization().getScale();
  double output_scale = output_type.getQuantization().getScale();

  const int trailing_dim = input_shape.rank() - 1;
  int excluding_last_dim = 1;
  for (int32_t i = 0; i < input_shape.rank() - 1; i++)
  {
    excluding_last_dim *= input_shape.dim(i);
  }
  const int last_dim = input_shape.dim(trailing_dim);

  const int32_t clamp_max = std::numeric_limits<uint8_t>::max();
  const int32_t clamp_min = std::numeric_limits<uint8_t>::min();

  uint8_t *input_data = reinterpret_cast<uint8_t *>(input.atOffset(0));

  float table[256];
  PopulateSoftmaxLookupTable(table, input_scale, 1.f);

  uint8_t *output_data = reinterpret_cast<uint8_t *>(output.atOffset(0));

  for (int i = 0; i < excluding_last_dim; ++i)
  {
    int32_t max_val = std::numeric_limits<uint8_t>::min();
    // Find max quantized value.
    for (int j = 0; j < last_dim; ++j)
    {
      max_val = std::max(max_val, static_cast<int32_t>(input_data[j]));
    }

    float sum_exp = 0.0f;
    const int32_t max_uint8 = std::numeric_limits<uint8_t>::max();
    const float *table_offset = &table[max_uint8 - max_val];
    // Calculate normalizer sum(exp(x)).
    for (int j = 0; j < last_dim; ++j)
    {
      sum_exp += table_offset[input_data[j]];
    }

    const float inv_sum_exp = 1.0f / (sum_exp * output_scale);
    // Normalize and quantize probabilities.
    for (int j = 0; j < last_dim; ++j)
    {
      const float prob_rescaled = table_offset[input_data[j]] * inv_sum_exp;
      const int32_t prob_quantized = static_cast<int32_t>(prob_rescaled + 0.5);
      output_data[j] =
        static_cast<uint8_t>(std::max(std::min(clamp_max, prob_quantized), clamp_min));
    }
    input_data += last_dim;
    output_data += last_dim;
  }
}

void Softmax(const mir::TensorVariant &arg, int axis, mir::TensorVariant &result)
{
  dispatch<SoftmaxImpl>(arg.getElementType(), arg, axis, result);
};

} // namespace mir_interpreter
