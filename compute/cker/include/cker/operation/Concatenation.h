/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __NNFW_CKER_CONCATENATION_H__
#define __NNFW_CKER_CONCATENATION_H__

#include "cker/Shape.h"
#include "cker/Types.h"

#include <cstdint>
#include <cmath>

namespace nnfw
{
namespace cker
{

template <typename Scalar>
inline void Concatenation(const ConcatenationParams &params, const Shape *const *input_shapes,
                          const Scalar *const *input_data, const Shape &output_shape,
                          Scalar *output_data)
{
  int axis = params.axis;
  int inputs_count = params.inputs_count;
  const int concat_dimensions = output_shape.DimensionsCount();
  assert(axis < concat_dimensions);

  int64_t concat_size = 0;
  for (int i = 0; i < inputs_count; i++)
  {
    assert(input_shapes[i]->DimensionsCount() == concat_dimensions);
    for (int j = 0; j < concat_dimensions; j++)
    {
      if (j != axis)
      {
        auto dim_checked = MatchingDim(*input_shapes[i], j, output_shape, j);
        UNUSED_RELEASE(dim_checked);
      }
    }
    concat_size += input_shapes[i]->Dims(axis);
  }
  assert(concat_size == output_shape.Dims(axis));
  int64_t outer_size = 1;
  for (int i = 0; i < axis; ++i)
  {
    outer_size *= output_shape.Dims(i);
  }
  // For all input arrays,
  // FlatSize() = outer_size * Dims(axis) * base_inner_size;
  int64_t base_inner_size = 1;
  for (int i = axis + 1; i < concat_dimensions; ++i)
  {
    base_inner_size *= output_shape.Dims(i);
  }

  Scalar *output_ptr = output_data;
  for (int k = 0; k < outer_size; k++)
  {
    for (int i = 0; i < inputs_count; ++i)
    {
      const int copy_size = input_shapes[i]->Dims(axis) * base_inner_size;
      memcpy(output_ptr, input_data[i] + k * copy_size, copy_size * sizeof(Scalar));
      output_ptr += copy_size;
    }
  }
}

// quantized as it takes scale as a floating point value. This should be fixed
// when optimizng this routine further.
inline void ConcatenationWithScaling(const ConcatenationParams &params,
                                     const Shape *const *input_shapes,
                                     const uint8_t *const *input_data, const Shape &output_shape,
                                     uint8_t *output_data)
{
  int axis = params.axis;
  const int32_t *input_zeropoint = params.input_zeropoint;
  const float *input_scale = params.input_scale;
  int inputs_count = params.inputs_count;
  const int32_t output_zeropoint = params.output_zeropoint;
  const float output_scale = params.output_scale;

  const int concat_dimensions = output_shape.DimensionsCount();
  assert(axis <= concat_dimensions);

  int64_t concat_size = 0;
  for (int i = 0; i < inputs_count; i++)
  {
    assert(input_shapes[i]->DimensionsCount() == concat_dimensions);
    for (int j = 0; j < concat_dimensions; j++)
    {
      if (j != axis)
      {
        assert(input_shapes[i]->Dims(j) == output_shape.Dims(j));
      }
    }
    concat_size += input_shapes[i]->Dims(axis);
  }
  assert(concat_size == output_shape.Dims(axis));
  int64_t outer_size = 1;
  for (int i = 0; i < axis; ++i)
  {
    outer_size *= output_shape.Dims(i);
  }
  // For all input arrays,
  // FlatSize() = outer_size * Dims(axis) * base_inner_size;
  int64_t base_inner_size = 1;
  for (int i = axis + 1; i < concat_dimensions; ++i)
  {
    base_inner_size *= output_shape.Dims(i);
  }

  const float inverse_output_scale = 1.f / output_scale;
  uint8_t *output_ptr = output_data;
  for (int k = 0; k < outer_size; k++)
  {
    for (int i = 0; i < inputs_count; ++i)
    {
      const int copy_size = input_shapes[i]->Dims(axis) * base_inner_size;
      const uint8_t *input_ptr = input_data[i] + k * copy_size;
      if (input_zeropoint[i] == output_zeropoint && input_scale[i] == output_scale)
      {
        memcpy(output_ptr, input_ptr, copy_size);
      }
      else
      {
        const float scale = input_scale[i] * inverse_output_scale;
        const float bias = -input_zeropoint[i] * scale;
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

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_CONCATENATION_H__
