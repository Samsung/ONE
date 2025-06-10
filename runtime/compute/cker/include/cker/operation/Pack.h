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

#ifndef __NNFW_CKER_PACK_H__
#define __NNFW_CKER_PACK_H__

#include "cker/Shape.h"
#include "cker/Types.h"

namespace nnfw
{
namespace cker
{

template <typename Scalar>
inline void Pack(const PackParams &params, const Scalar *const *input_data,
                 const Shape &output_shape, Scalar *output_data)
{
  const int dimensions = output_shape.DimensionsCount();
  int axis = params.axis;
  int inputs_count = params.inputs_count;

  int outer_size = 1;
  for (int i = 0; i < axis; i++)
  {
    outer_size *= output_shape.Dims(i);
  }
  int copy_size = 1;
  for (int i = params.axis + 1; i < dimensions; i++)
  {
    copy_size *= output_shape.Dims(i);
  }

  for (int i = 0; i < inputs_count; ++i)
  {
    for (int k = 0; k < outer_size; k++)
    {
      const Scalar *input_ptr = input_data[i] + copy_size * k;
      int loc = k * inputs_count * copy_size + i * copy_size;
      memcpy(output_data + loc, input_ptr, copy_size * sizeof(Scalar));
    }
  }
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_PACK_H__
