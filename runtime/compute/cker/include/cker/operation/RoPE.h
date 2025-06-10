/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __NNFW_CKER_ROPE_H__
#define __NNFW_CKER_ROPE_H__

#include <stdexcept>

#include "cker/Shape.h"
#include "cker/Types.h"
#include "cker/Utils.h"

namespace nnfw
{
namespace cker
{

template <typename T>
inline void RoPE(const RoPEMode mode, const Shape &input_shape, const T *input_data,
                 const Shape &sin_table_shape, const T *sin_table_data,
                 const Shape &cos_table_shape, const T *cos_table_data, const Shape &output_shape,
                 T *output_data)
{
  if (input_shape.Dims(3) != sin_table_shape.Dims(3))
    throw std::runtime_error("the dimension(3) of input and sin_table do not match");

  if (input_shape.Dims(3) != cos_table_shape.Dims(3))
    throw std::runtime_error("the dimension(3) of input and cos_table do not match");

  const int32_t i0_n = MatchingDim(input_shape, 0, output_shape, 0);
  const int32_t i1_n = MatchingDim(input_shape, 1, output_shape, 1);
  const int32_t i2_n = MatchingDim(input_shape, 2, output_shape, 2);
  const int32_t i3_n = MatchingDim(input_shape, 3, output_shape, 3);

  if (i3_n % 2 != 0)
    throw std::runtime_error("i3_n must be even number");

  if (mode == RoPEMode::kGptNeox)
  {
    for (int32_t i0 = 0; i0 < i0_n; ++i0)
    {
      for (int32_t i1 = 0; i1 < i1_n; ++i1)
      {
        for (int32_t i2 = 0; i2 < i2_n; ++i2)
        {
          for (int32_t i3 = 0; i3 < i3_n / 2; ++i3)
          {
            const int32_t offset = Offset(input_shape, i0, i1, i2, i3);
            const T x0 = input_data[offset];
            const T x1 = input_data[offset + i3_n / 2];

            output_data[offset] = x0 * cos_table_data[i3] - x1 * sin_table_data[i3];
            output_data[offset + i3_n / 2] =
              x0 * sin_table_data[i3 + i3_n / 2] + x1 * cos_table_data[i3 + i3_n / 2];
          }
        }
      }
    }
  }
  else
  {
    throw std::runtime_error("Unsupported RoPE mode");
  }
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_ROPE_H__
