/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __NNFW_CKER_SELECT_H__
#define __NNFW_CKER_SELECT_H__

#include "cker/Shape.h"

#include <cmath>

namespace nnfw
{
namespace cker
{

template <typename D, typename T>
void Select(const Shape &input_condition_shape, const D *input_condition_data,
            const Shape &input_x_shape, const T *input_x_data, const Shape &input_y_shape,
            const T *input_y_data, const Shape &output_shape, T *output_data)
{
  const int64_t flatsize =
      MatchingFlatSize(input_condition_shape, input_x_shape, input_y_shape, output_shape);
  for (int64_t i = 0; i < flatsize; ++i)
  {
    output_data[i] = (input_condition_data[i] != 0) ? input_x_data[i] : input_y_data[i];
  }
}

template <typename D, typename T>
void RankOneSelect(const Shape &input_condition_shape, const D *input_condition_data,
                   const Shape &input_x_shape, const T *input_x_data, const Shape &input_y_shape,
                   const T *input_y_data, const Shape &output_shape, T *output_data)
{
  const int64_t outer_size = input_condition_shape.FlatSize();
  assert(MatchingDim(input_x_shape, 0, input_y_shape, 0, output_shape, 0) == outer_size);
  const int64_t inner_size = MatchingFlatSizeSkipDim(input_x_shape, 0, input_y_shape, output_shape);

  int64_t offset = 0;
  for (int64_t i = 0; i < outer_size; i++)
  {
    const T *input_data = (input_condition_data[i] != 0) ? input_x_data : input_y_data;
    memcpy(output_data + offset, input_data + offset, inner_size * sizeof(T));
    offset += inner_size;
  }
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_SELECT_H__
