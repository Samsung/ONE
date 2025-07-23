/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __NNFW_CKER_DYNAMIC_UPDATE_SLICE_H__
#define __NNFW_CKER_DYNAMIC_UPDATE_SLICE_H__

#include "cker/Shape.h"

#include <vector>

namespace nnfw::cker
{

class DynamicUpdateSlice
{
public:
  DynamicUpdateSlice() = default;
  ~DynamicUpdateSlice() = default;

private:
  template <typename T>
  void UpdateSlice(int32_t current_dim, int32_t max_dim, const std::vector<int32_t> &output_stride,
                   const std::vector<int32_t> &update_stride, const Shape &update_shape,
                   const T *update, const std::vector<int64_t> &indices_data, T *output)
  {
    if (current_dim == max_dim)
      return;

    if (current_dim == max_dim - 1)
    {
      output += indices_data[current_dim] * output_stride[current_dim];
      memcpy(output, update, update_shape.Dims(max_dim - 1) * sizeof(T));
      return;
    }

    output += indices_data[current_dim] * output_stride[current_dim];
    for (int i = 0; i < update_shape.Dims(current_dim); ++i)
    {
      UpdateSlice(current_dim + 1, max_dim, output_stride, update_stride, update_shape, update,
                  indices_data, output);
      output += output_stride[current_dim];
      update += update_stride[current_dim];
    }
  }

public:
  template <typename T>
  void operator()(const Shape &input_shape, const T *input_data, const Shape &update_shape,
                  const T *update_data, const std::vector<int64_t> &indices_data, T *output_data)
  {
    // Special case 1 : output is copy of update
    if (input_shape == update_shape)
    {
      memcpy(output_data, update_data, update_shape.FlatSize() * sizeof(T));
      return;
    }

    // Prepare update
    if (input_data != output_data)
      memcpy(output_data, input_data, input_shape.FlatSize() * sizeof(T));

    // Special case 2: no update
    if (update_shape.FlatSize() == 0)
      return;

    // Calculate clamped_start_indices
    const auto input_dims = input_shape.DimensionsCount();
    std::vector<int64_t> clamped_start_indices(input_dims, 0);
    assert(input_dims == update_shape.DimensionsCount());
    for (int i = 0; i < input_dims; i++)
    {
      clamped_start_indices[i] = std::min<int64_t>(std::max<int64_t>(0, indices_data[i]),
                                                   input_shape.Dims(i) - update_shape.Dims(i));
    }

    // Calculate strides
    std::vector<int32_t> output_stride(input_dims);
    std::vector<int32_t> update_stride(input_dims);
    output_stride[input_dims - 1] = 1;
    update_stride[input_dims - 1] = 1;
    for (int i = input_dims - 2; i >= 0; --i)
    {
      output_stride[i] = output_stride[i + 1] * input_shape.Dims(i + 1);
      update_stride[i] = update_stride[i + 1] * update_shape.Dims(i + 1);
    }

    UpdateSlice<T>(0, input_dims, output_stride, update_stride, update_shape, update_data,
                   clamped_start_indices, output_data);
  }
};

} // namespace nnfw::cker

#endif // __NNFW_CKER_DYNAMIC_UPDATE_SLICE_H__
