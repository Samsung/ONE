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

#ifndef __NNFW_CKER_UNPACK_H__
#define __NNFW_CKER_UNPACK_H__

#include "cker/Shape.h"
#include "cker/Types.h"

namespace nnfw
{
namespace cker
{

template <typename Scalar>
void Unpack(const UnpackParams &params, const Shape &input_shape, const Scalar *input_data,
            [[maybe_unused]] const Shape &output_shape, Scalar *const *output_datas)
{
  const int dimensions = input_shape.DimensionsCount();
  const int outputs_count = params.num_split;

  int outer_size = 1;
  for (int i = 0; i < params.axis; i++)
  {
    outer_size *= input_shape.Dims(i);
  }
  int copy_size = 1;
  for (int i = params.axis + 1; i < dimensions; i++)
  {
    copy_size *= input_shape.Dims(i);
  }
  assert(output_shape.FlatSize() == copy_size * outer_size);

  for (int i = 0; i < outputs_count; ++i)
  {
    for (int k = 0; k < outer_size; k++)
    {
      Scalar *output_ptr = output_datas[i] + copy_size * k;
      int loc = k * outputs_count * copy_size + i * copy_size;
      memcpy(output_ptr, input_data + loc, copy_size * sizeof(Scalar));
    }
  }
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_UNPACK_H__
