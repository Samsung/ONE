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

#ifndef __NNFW_CKER_REVERSE_H__
#define __NNFW_CKER_REVERSE_H__

#include "cker/Shape.h"
#include "cker/Types.h"
#include "cker/Utils.h"

namespace nnfw
{
namespace cker
{

template <typename Scalar>
void Reverse(int axis, const Shape &input_shape, const Scalar *input_data, const Shape &,
             Scalar *output_data)
{
  int outer_size = 1;
  for (int i = 0; i < axis; ++i)
  {
    outer_size *= input_shape.Dims(i);
  }

  int copy_size = 1;
  for (int i = axis + 1; i < input_shape.DimensionsCount(); ++i)
  {
    copy_size *= input_shape.Dims(i);
  }

  const int dims_at_axis = input_shape.Dims(axis);
  for (int i = 0; i < outer_size; ++i)
  {
    for (int j = 0; j < dims_at_axis; ++j)
    {
      const int start_pos = (i * dims_at_axis + j) * copy_size;
      Scalar *output_ptr = output_data + start_pos;
      int loc = (i * dims_at_axis + dims_at_axis - j - 1) * copy_size;
      memcpy(output_ptr, input_data + loc, copy_size * sizeof(Scalar));
    }
  }
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_REVERSE_H__
