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

#include "Concat.h"
#include "Common.h"

#include <cmath>
#include <cstring>

namespace mir_interpreter
{

template <typename T> struct ConcatImpl
{
  static void run(const std::vector<std::reference_wrapper<const mir::TensorVariant>> &inputs,
                  int axis, mir::TensorVariant &output);
};

template <typename T>
void ConcatImpl<T>::run(const std::vector<std::reference_wrapper<const mir::TensorVariant>> &inputs,
                        int axis, mir::TensorVariant &output)
{
  const auto &output_shape = output.getShape();
  const size_t inputs_count = inputs.size();
  const int32_t concat_dims = output_shape.rank();
  int64_t concat_size = 0;
  for (size_t i = 0; i < inputs_count; i++)
  {
    const auto &input_shape = inputs[i].get().getShape();
    assert(input_shape.rank() == concat_dims);
    for (int32_t j = 0; j < concat_dims; j++)
    {
      if (j != axis)
      {
        assert(input_shape.dim(j) == output_shape.dim(j));
      }
    }
    concat_size += input_shape.dim(axis);
  }
  assert(concat_size == output_shape.dim(axis));
  // Outer size before axis
  int32_t outer_size = 1;
  for (int32_t i = 0; i < axis; i++)
    outer_size *= output_shape.dim(i);
  // Inner size after axis
  int32_t base_inner_size = 1;
  for (int32_t i = axis + 1; i < concat_dims; i++)
    base_inner_size *= output_shape.dim(i);
  // flatten = outer_size * dim(axis) * base_inner_size;
  std::vector<int32_t> copy_sizes;
  std::vector<char *> input_ptrs;
  for (size_t i = 0; i < inputs_count; i++)
  {
    const auto input_shape = inputs[i].get().getShape();
    copy_sizes.push_back(input_shape.dim(axis) * base_inner_size);
    input_ptrs.push_back(inputs[i].get().atOffset(0));
  }

  char *output_ptr = output.atOffset(0);
  const size_t elem_size = inputs[0].get().getElementSize();
  for (int32_t i = 0; i < outer_size; i++)
  {
    for (size_t j = 0; j < inputs_count; j++)
    {
      std::memcpy(output_ptr, input_ptrs[j], copy_sizes[j] * elem_size);
      output_ptr += copy_sizes[j] * elem_size;
      input_ptrs[j] += copy_sizes[j] * elem_size;
    }
  }
}

template <> struct ConcatImpl<uint8_t>
{
  static void run(const std::vector<std::reference_wrapper<const mir::TensorVariant>> &inputs,
                  int axis, mir::TensorVariant &output);
};

void ConcatImpl<uint8_t>::run(
  const std::vector<std::reference_wrapper<const mir::TensorVariant>> &inputs, int axis,
  mir::TensorVariant &output)
{
  const size_t inputs_count = inputs.size();
  std::vector<int32_t> input_zeropoints(inputs_count);
  std::vector<float> input_scales(inputs_count);
  const auto &output_shape = output.getShape();
  const int32_t concat_dimensions = output_shape.rank();
  int64_t concat_size = 0;
  for (size_t i = 0; i < inputs_count; i++)
  {
    const auto &input_type = inputs[i].get().getType();
    assert(input_type.isQuantized());
    assert(input_type.getElementType() == mir::DataType::UINT8);
    const auto &input_shape = input_type.getShape();
    assert(input_shape.rank() == concat_dimensions);

    for (int32_t j = 0; j < concat_dimensions; j++)
      if (j != axis)
        assert(input_shape.dim(j) == output_shape.dim(j));

    concat_size += input_shape.dim(axis);
    input_zeropoints[i] = input_type.getQuantization().getZeroPoint();
    input_scales[i] = input_type.getQuantization().getScale();
  }
  assert(concat_size == output_shape.dim(axis));

  const auto &output_type = output.getType();
  assert(output_type.isQuantized());
  int32_t output_zeropoint = output_type.getQuantization().getZeroPoint();
  float output_scale = output_type.getQuantization().getScale();

  // Outer size before axis
  int32_t outer_size = 1;
  for (int32_t i = 0; i < axis; i++)
    outer_size *= output_shape.dim(i);
  // Inner size after axis
  int32_t base_inner_size = 1;
  for (int32_t i = axis + 1; i < concat_dimensions; i++)
    base_inner_size *= output_shape.dim(i);
  // flatten = outer_size * dim(axis) * base_inner_size;

  uint8_t *output_ptr = reinterpret_cast<uint8_t *>(output.atOffset(0));

  const float inverse_output_scale = 1.f / output_scale;
  for (int k = 0; k < outer_size; k++)
  {
    for (size_t i = 0; i < inputs_count; ++i)
    {
      const mir::TensorVariant &input = inputs[i];
      const int copy_size = input.getShape().dim(axis) * base_inner_size;
      const char *input_data = input.atOffset(0) + k * copy_size;
      const uint8_t *input_ptr = reinterpret_cast<const uint8_t *>(input_data);
      if (input_zeropoints[i] == output_zeropoint && input_scales[i] == output_scale)
      {
        std::memcpy(output_ptr, input_ptr, copy_size);
      }
      else
      {
        const float scale = input_scales[i] * inverse_output_scale;
        const float bias = -input_zeropoints[i] * scale;
        for (int j = 0; j < copy_size; ++j)
        {
          const int32_t value =
            static_cast<int32_t>(std::round(input_ptr[j] * scale + bias)) + output_zeropoint;
          output_ptr[j] = static_cast<uint8_t>(std::max(std::min(255, value), 0));
        }
      }
      output_ptr += copy_size;
    }
  }
}

void Concat(const std::vector<std::reference_wrapper<const mir::TensorVariant>> &inputs, int axis,
            mir::TensorVariant &output)
{
  dispatch<ConcatImpl>(inputs[0].get().getElementType(), inputs, axis, output);
}

} // namespace mir_interpreter
