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

#ifndef __NNFW_CKER_ONEHOT_H__
#define __NNFW_CKER_ONEHOT_H__

#include "cker/Shape.h"

namespace nnfw
{
namespace cker
{

template <typename T, typename TI>
void OneHot(const int32_t depth, const T on_value, const T off_value, int32_t axis,
            const Shape &indices_shape, const TI *indices_data, const Shape &, T *output_data)
{
  if (axis == -1)
    axis = indices_shape.DimensionsCount();

  // prefix_dim_size == # of elements before the axis
  // depth == # of elements per axis
  // suffix_dim_size == # of elements after the axis
  int prefix_dim_size = 1;
  for (int i = 0; i < axis; ++i)
  {
    prefix_dim_size *= indices_shape.Dims(i);
  }
  const int suffix_dim_size = indices_shape.FlatSize() / prefix_dim_size;

  // View the indices as a matrix of size:
  //     prefix_dim_size x suffix_dim_size
  // View the output as a matrix of size:
  //     prefix_dim_size x depth x suffix_dim_size
  // Then the output is:
  //     output(i, j, k) == (indices(i, k) == j) ? on : off
  for (int i = 0; i < prefix_dim_size; ++i)
  {
    for (int j = 0; j < depth; ++j)
    {
      for (int k = 0; k < suffix_dim_size; ++k, ++output_data)
      {
        *output_data =
          static_cast<int>(indices_data[i * suffix_dim_size + k]) == j ? on_value : off_value;
      }
    }
  }
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_ONEHOT_H__
