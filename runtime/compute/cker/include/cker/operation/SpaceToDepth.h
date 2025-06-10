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

#ifndef __NNFW_CKER_SPACE_TO_DEPTH_H__
#define __NNFW_CKER_SPACE_TO_DEPTH_H__

#include "cker/Shape.h"
#include "cker/Types.h"

namespace nnfw
{
namespace cker
{

template <typename T>
inline void SpaceToDepth(const SpaceToDepthParams &params, const Shape &unextended_input_shape,
                         const T *input_data, const Shape &unextended_output_shape, T *output_data)
{
  assert(unextended_input_shape.DimensionsCount() <= 4);
  assert(unextended_output_shape.DimensionsCount() <= 4);
  const Shape input_shape = Shape::ExtendedShape(4, unextended_input_shape);
  const Shape output_shape = Shape::ExtendedShape(4, unextended_output_shape);

  const int output_depth = output_shape.Dims(3);
  const int output_width = output_shape.Dims(2);
  const int output_height = output_shape.Dims(1);

  const int input_depth = input_shape.Dims(3);
  const int batch_size = input_shape.Dims(0);

  // Number of continuous values that we can copy in one interation.
  const int stride = params.block_size * input_depth;

  for (int batch = 0; batch < batch_size; ++batch)
  {
    for (int out_h = 0; out_h < output_height; ++out_h)
    {
      T *output_ptr = output_data + Offset(output_shape, batch, out_h, 0, 0);
      for (int offset_h = 0; offset_h < params.block_size; ++offset_h)
      {
        T *dst = output_ptr;
        for (int out_w = 0; out_w < output_width; ++out_w)
        {
          memcpy(dst, input_data, stride * sizeof(T));
          input_data += stride;
          dst += output_depth;
        }
        output_ptr += stride;
      }
    }
  }
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_SPACE_TO_DEPTH_H__
