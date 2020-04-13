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

#ifndef __NNFW_CKER_GATHER_H__
#define __NNFW_CKER_GATHER_H__

#include "cker/Shape.h"
#include "cker/Types.h"
#include "cker/Utils.h"

namespace nnfw
{
namespace cker
{

template <typename T, typename CoordsT = int32_t>
inline void Gather(const GatherParams &op_params, const Shape &input_shape, const T *input_data,
                   const Shape &coords_shape, const CoordsT *coords_data, const Shape &,
                   T *output_data)
{
  int axis = op_params.axis;
  if (axis < 0)
  {
    axis += input_shape.DimensionsCount();
  }
  assert(axis >= 0);
  assert(axis < input_shape.DimensionsCount());
  const int axis_size = input_shape.Dims(axis);
  const int coords_count = coords_shape.FlatSize();

  int outer_size = 1;
  for (int i = 0; i < axis; ++i)
  {
    outer_size *= input_shape.Dims(i);
  }

  int inner_size = 1;
  for (int i = axis + 1; i < input_shape.DimensionsCount(); ++i)
  {
    inner_size *= input_shape.Dims(i);
  }

  for (int outer = 0; outer < outer_size; ++outer)
  {
    for (int i = 0; i < coords_count; ++i)
    {
      assert(coords_data[i] >= 0);
      assert(coords_data[i] < axis_size);
      std::memcpy(output_data + (outer * coords_count + i) * inner_size,
                  input_data + (outer * axis_size + coords_data[i]) * inner_size,
                  sizeof(T) * inner_size);
    }
  }
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_GATHER_H__
