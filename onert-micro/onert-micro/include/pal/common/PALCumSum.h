/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2021 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ONERT_MICRO_PAL_CUM_SUM_COMMON_H
#define ONERT_MICRO_PAL_CUM_SUM_COMMON_H

#include "core/OMKernelData.h"
#include "core/OMRuntimeShape.h"

using namespace onert_micro::core;

namespace onert_micro::execute::pal
{

template <typename T>
OMStatus CumSum(const CumSumParams &params, const OMRuntimeShape &input_shape, const T *input_data,
                int32_t axis, T *output_data)
{
  const int32_t rank = input_shape.dimensionsCount();

  int32_t inner_size = 1;
  int32_t outer_size = 1;
  int32_t depth_size = 1;

  for (int32_t i = 0; i < rank; ++i)
  {
    if (i < axis)
      inner_size *= input_shape.dims(i);

    else if (i > axis)
      outer_size *= input_shape.dims(i);

    else
      depth_size *= input_shape.dims(i);
  }

  // clang-format off

  auto get_adj_index_fn = [reverse = params.reverse](int32_t index, int32_t dims)
  {
    if (!reverse)
      return index;

    return (dims - 1) - index;
  };

  // clang-format on

  for (int32_t outer = 0; outer < outer_size; ++outer)
  {
    int32_t outer_adj = get_adj_index_fn(outer, outer_size);

    for (int32_t inner = 0; inner < inner_size; ++inner)
    {
      T accumulator = 0;
      int32_t inner_adj = get_adj_index_fn(inner, inner_size);

      for (int32_t depth = 0; depth < depth_size; ++depth)
      {
        int32_t depth_adj = get_adj_index_fn(depth, depth_size);
        int32_t index = outer_adj;

        index += inner_adj * depth_size * outer_size;
        index += depth_adj * outer_size;

        if (params.exclusive)
        {
          output_data[index] = accumulator;
          accumulator += input_data[index];

          continue;
        }

        accumulator += input_data[index];
        output_data[index] = accumulator;
      }
    }
  }

  return Ok;
}

} // namespace onert_micro::execute::pal

#endif // ONERT_MICRO_PAL_CUM_SUM_COMMON_H
