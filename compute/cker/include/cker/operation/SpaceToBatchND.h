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

#ifndef __NNFW_CKER_SPACE_TO_BATCH_ND_H__
#define __NNFW_CKER_SPACE_TO_BATCH_ND_H__

#include "cker/Shape.h"
#include "cker/Types.h"

namespace nnfw
{
namespace cker
{

template <typename T>
inline void SpaceToBatchND(const SpaceToBatchParams &params, const Shape &unextended_input_shape,
                           const T *input_data, const Shape &unextended_block_shape_shape,
                           const int32_t *block_shape_data, const Shape &unextended_padding_shape,
                           const int32_t *paddings_data, const Shape &unextended_output_shape,
                           T *output_data)
{
  UNUSED_RELEASE(unextended_block_shape_shape);
  UNUSED_RELEASE(unextended_padding_shape);

  assert(unextended_input_shape.DimensionsCount() <= 4);
  assert(unextended_output_shape.DimensionsCount() <= 4);
  const Shape input_shape = Shape::ExtendedShape(4, unextended_input_shape);
  const Shape output_shape = Shape::ExtendedShape(4, unextended_output_shape);

  const int depth = input_shape.Dims(3);
  const int input_width = input_shape.Dims(2);
  const int input_height = input_shape.Dims(1);
  const int input_batch_size = input_shape.Dims(0);

  const int output_width = output_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_batch_size = output_shape.Dims(0);

  const int block_shape_height = block_shape_data[0];
  const int block_shape_width = block_shape_data[1];
  const int padding_top = paddings_data[0];
  const int padding_left = paddings_data[2];

  // For uint8 quantized, the correct padding "zero value" is the output offset.
  const int32_t pad_value = params.output_offset;

  for (int out_b = 0; out_b < output_batch_size; ++out_b)
  {
    int input_batch = out_b % input_batch_size;
    int shift_w = (out_b / input_batch_size) % block_shape_width;
    int shift_h = (out_b / input_batch_size) / block_shape_width;
    for (int out_h = 0; out_h < output_height; ++out_h)
    {
      for (int out_w = 0; out_w < output_width; ++out_w)
      {
        T *out = output_data + Offset(output_shape, out_b, out_h, out_w, 0);
        if (out_h * block_shape_height + shift_h < padding_top ||
            out_h * block_shape_height + shift_h >= padding_top + input_height ||
            out_w * block_shape_width + shift_w < padding_left ||
            out_w * block_shape_width + shift_w >= padding_left + input_width)
        {
          // This may not execute correctly when pad_value != 0 and T != uint8.
          memset(out, pad_value, depth * sizeof(T));
        }
        else
        {
          const T *in =
            input_data + Offset(input_shape, input_batch,
                                (out_h * block_shape_height + shift_h) - padding_top,
                                (out_w * block_shape_width + shift_w) - padding_left, 0);
          memcpy(out, in, depth * sizeof(T));
        }
      }
    }
  }
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_SPACE_TO_BATCH_ND_H__
