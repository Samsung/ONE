/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __NNFW_CKER_FLOOR_DIV_H__
#define __NNFW_CKER_FLOOR_DIV_H__

#include "cker/Shape.h"
#include "cker/Utils.h"

namespace nnfw
{
namespace cker
{

template <typename T>
inline void FloorDivBroadcast(const Shape &unextended_input1_shape, const T *input1_data,
                              const Shape &unextended_input2_shape, const T *input2_data,
                              const Shape &unextended_output_shape, T *output_data)
{
  assert(unextended_input1_shape.DimensionsCount() <= 4);
  assert(unextended_input2_shape.DimensionsCount() <= 4);
  assert(unextended_output_shape.DimensionsCount() <= 4);
  const Shape output_shape = Shape::ExtendedShape(4, unextended_output_shape);

  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(unextended_input1_shape, unextended_input2_shape, &desc1,
                                      &desc2);

  for (int b = 0; b < output_shape.Dims(0); ++b)
  {
    for (int y = 0; y < output_shape.Dims(1); ++y)
    {
      for (int x = 0; x < output_shape.Dims(2); ++x)
      {
        for (int c = 0; c < output_shape.Dims(3); ++c)
        {
          auto out_idx = Offset(output_shape, b, y, x, c);
          auto in1_idx = SubscriptToIndex(desc1, b, y, x, c);
          auto in2_idx = SubscriptToIndex(desc2, b, y, x, c);
          auto in1_val = input1_data[in1_idx];
          auto in2_val = input2_data[in2_idx];
          output_data[out_idx] = std::floor(
            std::divides<double>()(static_cast<double>(in1_val), static_cast<double>(in2_val)));
        }
      }
    }
  }
}

template <typename T>
inline void FloorDivElementwise(const Shape &shape, const T *input1_data, const T *input2_data,
                                T *output_data)
{

  int num_elements = shape.FlatSize();

  for (int t = 0; t < num_elements; t++)
  {
    output_data[t] = std::floor(std::divides<double>()(static_cast<double>(input1_data[t]),
                                                       static_cast<double>(input2_data[t])));
  }
}

} // namespace cker

} // namespace nnfw
#endif
