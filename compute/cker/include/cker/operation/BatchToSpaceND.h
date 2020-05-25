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

#ifndef __NNFW_CKER_BATCHTOSPACEND_H__
#define __NNFW_CKER_BATCHTOSPACEND_H__

#include "cker/Shape.h"

namespace nnfw
{
namespace cker
{

inline void GetIndexRange(int spatial_index_dim, int block_shape_dim, int input_dim, int output_dim,
                          int *start_index, int *end_index)
{
  *start_index = std::max(0, (-spatial_index_dim + block_shape_dim - 1) / block_shape_dim);
  *end_index =
      std::min(input_dim, (output_dim - spatial_index_dim + block_shape_dim - 1) / block_shape_dim);
}

inline void tf_assert(bool cond)
{
  if (!(cond))
    throw std::runtime_error("Input/ouput condition fail");
}

template <typename T>
inline void BatchToSpaceND(const Shape &unextended_input1_shape, const T *input1_data,
                           const Shape &unextended_input2_shape, const int32_t *block_shape_data,
                           const Shape &unextended_input3_shape, const int32_t *crops_data,
                           const Shape &unextended_output_shape, T *output_data)
{

  (void)unextended_input2_shape;
  (void)unextended_input3_shape;

  tf_assert(unextended_input1_shape.DimensionsCount() >= 3);
  tf_assert(unextended_input1_shape.DimensionsCount() <= 4);
  tf_assert(unextended_input1_shape.DimensionsCount() == unextended_output_shape.DimensionsCount());

  // Extends the input/output shape from 3D to 4D if needed, NHC -> NH1C.
  auto extend_shape = [](const Shape &shape) {

    if (shape.DimensionsCount() == 4)
    {
      return shape;
    }
    Shape new_shape(4, 1);
    new_shape.SetDim(0, shape.Dims(0));
    new_shape.SetDim(1, shape.Dims(1));
    new_shape.SetDim(3, shape.Dims(2));
    return new_shape;
  };

  const Shape input1_shape = extend_shape(unextended_input1_shape);
  const Shape output_shape = extend_shape(unextended_output_shape);

  const int output_width = output_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_batch_size = output_shape.Dims(0);

  const int depth = input1_shape.Dims(3);
  const int input_width = input1_shape.Dims(2);
  const int input_height = input1_shape.Dims(1);
  const int input_batch_size = input1_shape.Dims(0);

  const int block_shape_height = block_shape_data[0];
  const int block_shape_width =
      unextended_input1_shape.DimensionsCount() == 4 ? block_shape_data[1] : 1;
  const int crops_top = crops_data[0];
  const int crops_left = unextended_input1_shape.DimensionsCount() == 4 ? crops_data[2] : 0;

  for (int in_batch = 0; in_batch < input_batch_size; ++in_batch)
  {
    const int out_batch = in_batch % output_batch_size;
    const int spatial_offset = in_batch / output_batch_size;

    int in_h_start = 0;
    int in_h_end = 0;
    // GetIndexRange ensures start and end indices are in [0, output_height).
    GetIndexRange(spatial_offset / block_shape_width - crops_top, block_shape_height, input_height,
                  output_height, &in_h_start, &in_h_end);

    for (int in_h = in_h_start; in_h < in_h_end; ++in_h)
    {
      const int out_h = in_h * block_shape_height + spatial_offset / block_shape_width - crops_top;
      tf_assert(out_h >= 0);
      tf_assert(out_h < output_height);

      int in_w_start = 0;
      int in_w_end = 0;
      // GetIndexRange ensures start and end indices are in [0, output_width).
      GetIndexRange(spatial_offset % block_shape_width - crops_left, block_shape_width, input_width,
                    output_width, &in_w_start, &in_w_end);

      for (int in_w = in_w_start; in_w < in_w_end; ++in_w)
      {
        const int out_w =
            in_w * block_shape_width + spatial_offset % block_shape_width - crops_left;
        tf_assert(out_w >= 0);
        tf_assert(out_w < output_width);
        T *out = output_data + Offset(output_shape, out_batch, out_h, out_w, 0);
        const T *in = input1_data + Offset(input1_shape, in_batch, in_h, in_w, 0);
        memcpy(out, in, depth * sizeof(T));
      }
    }
  }
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_BATCHTOSPACEND_H__
