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

#ifndef __NNFW_CKER_TILE_H__
#define __NNFW_CKER_TILE_H__

#include "cker/Shape.h"

namespace nnfw
{
namespace cker
{

template <typename T, typename M>
void CopyMultipleTimes(const T *in_data, int32_t in_size, M multiplier, T *out_data)
{
  for (M i = 0; i < multiplier; ++i)
  {
    const T *in_end = in_data + in_size;
    T *new_out_data = std::copy(in_data, in_end, out_data);
    in_data = out_data;
    out_data = new_out_data;
  }
}

template <typename T, typename M>
std::pair<int, int> TileOneDimension(const Shape &in_dimensions, const T *in_data,
                                     const M *multipliers, T *out_data, int dimension)
{
  const int dimension_size = in_dimensions.Dims(dimension);
  if (dimension == in_dimensions.DimensionsCount() - 1)
  {
    CopyMultipleTimes(in_data, dimension_size, multipliers[dimension], out_data);
    return std::make_pair(dimension_size,
                          dimension_size * static_cast<int>(multipliers[dimension]));
  }
  int total_stride_size = 0, total_tiled_stride_size = 0;
  const T *copy_from_data = in_data;
  T *copy_to_data = out_data;
  for (int i = 0; i < dimension_size; ++i)
  {
    int stride_size = 0, tiled_stride_size = 0;
    std::tie(stride_size, tiled_stride_size) =
      TileOneDimension(in_dimensions, copy_from_data, multipliers, copy_to_data, dimension + 1);
    copy_from_data += stride_size;
    copy_to_data += tiled_stride_size;
    total_stride_size += stride_size;
    total_tiled_stride_size += tiled_stride_size;
  }
  CopyMultipleTimes(out_data, total_tiled_stride_size, multipliers[dimension] - 1,
                    out_data + total_tiled_stride_size);
  return std::make_pair(total_stride_size,
                        static_cast<int>(total_tiled_stride_size * multipliers[dimension]));
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_TILE_H__
