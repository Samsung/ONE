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

#ifndef __NNFW_CKER_SPLIT_V_H__
#define __NNFW_CKER_SPLIT_V_H__

#include "cker/Shape.h"
#include "cker/Types.h"

namespace nnfw
{
namespace cker
{

template <typename Scalar>
void SplitV(const SplitVParams &params, const Shape &input_shape, const Scalar *input_data,
            std::vector<nnfw::cker::Shape> &output_shapes, Scalar *const *output_data)
{
  const int split_dimensions = input_shape.DimensionsCount();
  int axis = params.axis < 0 ? params.axis + split_dimensions : params.axis;
  int outputs_count = params.num_split;

  int64_t split_size = 0;

  for (int i = 0; i < outputs_count; i++)
  {
    // TFLITE_DCHECK_EQ(output_shapes[i]->DimensionsCount(), split_dimensions);
    for (int j = 0; j < split_dimensions; j++)
    {
      if (j != axis)
      {
        MatchingDim(output_shapes[i], j, input_shape, j);
      }
    }
    split_size += output_shapes[i].Dims(axis);
  }

  int64_t outer_size = 1;
  for (int i = 0; i < axis; ++i)
  {
    outer_size *= input_shape.Dims(i);
  }
  // For all output arrays,
  // FlatSize() = outer_size * Dims(axis) * base_inner_size;
  int64_t base_inner_size = 1;
  for (int i = axis + 1; i < split_dimensions; ++i)
  {
    base_inner_size *= input_shape.Dims(i);
  }

  const Scalar *input_ptr = input_data;
  int copy_size = 0;
  for (int k = 0; k < outer_size; k++)
  {
    for (int i = 0; i < outputs_count; ++i)
    {
      copy_size = output_shapes[i].Dims(axis) * base_inner_size;
      memcpy(output_data[i] + k * copy_size, input_ptr, copy_size * sizeof(Scalar));
      input_ptr += copy_size;
    }
  }
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_SPLIT_V_H__
