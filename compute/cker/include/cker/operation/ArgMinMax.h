/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __NNFW_CKER_ARGMINMAX_H__
#define __NNFW_CKER_ARGMINMAX_H__

#include "cker/Shape.h"

namespace nnfw
{
namespace cker
{

template <typename T1, typename T2, typename Cmp>
void ArgMinMax(const Shape &input1_shape, const T1 *input1_data,
               [[maybe_unused]] const Shape &output_shape, T2 *output_data, int32_t axis,
               const Cmp &cmp)
{
  assert(input1_shape.DimensionsCount() > 0);
  assert(input1_shape.DimensionsCount() - 1 == output_shape.DimensionsCount());
  if (axis < 0)
  {
    axis += input1_shape.DimensionsCount();
  }
  const int axis_size = input1_shape.Dims(axis);

  int outer_size = 1;
  for (int i = 0; i < axis; ++i)
  {
    assert(input1_shape.Dims(i) == output_shape.Dims(i));
    outer_size *= input1_shape.Dims(i);
  }

  int inner_size = 1;
  const int dims_count = input1_shape.DimensionsCount();
  for (int i = axis + 1; i < dims_count; ++i)
  {
    assert(input1_shape.Dims(i) == output_shape.Dims(i - 1));
    inner_size *= input1_shape.Dims(i);
  }
  for (int outer = 0; outer < outer_size; ++outer)
  {
    for (int inner = 0; inner < inner_size; ++inner)
    {
      auto min_max_value = input1_data[outer * axis_size * inner_size + inner];
      T2 min_max_index = 0;
      for (int i = 1; i < axis_size; ++i)
      {
        const auto &curr_value = input1_data[(outer * axis_size + i) * inner_size + inner];
        if (cmp(curr_value, min_max_value))
        {
          min_max_value = curr_value;
          min_max_index = static_cast<T2>(i);
        }
      }
      output_data[outer * inner_size + inner] = min_max_index;
    }
  }
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_ARGMINMAX_H__
