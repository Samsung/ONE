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

#ifndef ONERT_MICRO_EXECUTE_PAL_MEAN_COMMON_H
#define ONERT_MICRO_EXECUTE_PAL_MEAN_COMMON_H

#include "core/OMKernelData.h"
#include "PALUtils.h"

#include <cmath>

namespace onert_micro
{
namespace execute
{
namespace pal
{

namespace
{
// This method parses the input 'axis' to remove duplicates and handle negative
// values, and returns a valid 'out_axis'
inline bool resolveAxis(const int num_dims, const int *axis, const int64_t num_axis, int *out_axis,
                        int *out_num_axis)
{
  *out_num_axis = 0; // Just in case.
  // Short-circuit axis resolution for scalars; the axis will go unused.
  if (num_dims == 0)
  {
    return true;
  }
  // o(n^2) is fine since out_num_axis should be really small, mostly <= 4
  for (int64_t idx = 0; idx < num_axis; ++idx)
  {
    // Handle negative index. A positive index 'p_idx' can be represented as a
    // negative index 'n_idx' as: n_idx = p_idx-num_dims
    // eg: For num_dims=3, [0, 1, 2] is the same as [-3, -2, -1]  */
    int current = axis[idx] < 0 ? (axis[idx] + num_dims) : axis[idx];
    if (current < 0 || current >= num_dims)
    {
      return false;
    }
    bool is_dup = false;
    for (int j = 0; j < *out_num_axis; ++j)
    {
      if (out_axis[j] == current)
      {
        is_dup = true;
        break;
      }
    }
    if (!is_dup)
    {
      out_axis[*out_num_axis] = current;
      *out_num_axis += 1;
    }
  }
  return true;
}

// A generic reduce method that can be used for reduce_sum, reduce_mean, etc.
// This method iterates through input data and reduce elements along the
// dimensions given in axis.
template <typename In, typename Out>
inline bool reduce(const In *input_data, const int *input_dims, const int *,
                   const int input_num_dims, const int, const int *axis, const int num_axis,
                   int *input_iter, Out reducer(Out, const In), Out *output_data)
{
  // Reset input iterator.
  for (int idx = 0; idx < input_num_dims; ++idx)
  {
    input_iter[idx] = 0;
  }
  // Iterate through input_data.
  do
  {
    size_t input_offset = reducedOutputOffset(input_num_dims, input_dims, input_iter, 0, nullptr);
    size_t output_offset =
      reducedOutputOffset(input_num_dims, input_dims, input_iter, num_axis, axis);
    output_data[output_offset] = reducer(output_data[output_offset], input_data[input_offset]);
  } while (nextIndex(input_num_dims, input_dims, input_iter));
  return true;
}

// This method expects that output_data has been initialized.
template <typename In, typename Out>
inline bool reduceSumImpl(const In *input_data, const int *input_dims, const int *output_dims,
                          const int input_num_dims, const int output_num_dims, const int *axis,
                          const int num_axis, int *input_iter, Out *output_data)
{
  auto reducer = [](const Out current, const In in) -> Out {
    const Out actual_in = static_cast<Out>(in);
    return current + actual_in;
  };
  return reduce<In, Out>(input_data, input_dims, output_dims, input_num_dims, output_num_dims, axis,
                         num_axis, input_iter, reducer, output_data);
}
} // namespace

template <typename T, typename U>
OMStatus Mean(const T *input_data, const core::OMRuntimeShape &input_shape, T *output_data,
              const core::OMRuntimeShape &output_shape, const int *axis,
              const int num_axis_dimensions, bool, int *temp_index, int *resolved_axis, U *temp_sum)
{
  const auto output_num_dims = output_shape.dimensionsCount();
  const auto *output_dims = output_shape.dimsData();
  // Reset output data.
  size_t num_outputs = 1;
  for (int idx = 0; idx < output_num_dims; ++idx)
  {
    auto current = static_cast<size_t>(output_dims[idx]);
    // Overflow prevention.
    if (num_outputs > std::numeric_limits<size_t>::max() / current)
    {
      return UnknownError;
    }
    num_outputs *= current;
  }
  for (size_t idx = 0; idx < num_outputs; ++idx)
  {
    output_data[idx] = T();
    temp_sum[idx] = U();
  }

  const auto input_num_dims = input_shape.dimensionsCount();
  const auto *input_dims = input_shape.dimsData();
  // Resolve axis.
  int num_resolved_axis = 0;
  if (!resolveAxis(input_num_dims, axis, num_axis_dimensions, resolved_axis, &num_resolved_axis))
  {
    return UnknownError;
  }

  if (!reduceSumImpl<T, U>(input_data, input_dims, output_dims, input_num_dims, output_num_dims,
                           resolved_axis, num_resolved_axis, temp_index, temp_sum))
  {
    return UnknownError;
  }

  // Calculate mean by dividing output_data by num of aggregated element.
  size_t num_elements_in_axis = 1;
  for (int idx = 0; idx < num_resolved_axis; ++idx)
  {
    auto current = static_cast<size_t>(input_dims[resolved_axis[idx]]);
    // Overflow prevention.
    if (current > (std::numeric_limits<size_t>::max() / num_elements_in_axis))
    {
      return UnknownError;
    }
    num_elements_in_axis *= current;
  }

  if (num_elements_in_axis > 0)
  {
    for (size_t idx = 0; idx < num_outputs; ++idx)
    {
      output_data[idx] = static_cast<T>(temp_sum[idx] / static_cast<U>(num_elements_in_axis));
    }
  }
  return Ok;
}

OMStatus Mean(const core::MeanParams &op_params, const core::OMRuntimeShape &unextended_input_shape,
              const float *input_data, const core::OMRuntimeShape &unextended_output_shape,
              float *output_data)
{
  // Current implementation only supports dimension equals 4 and simultaneous
  // reduction over width and height.
  const core::OMRuntimeShape input_shape =
    core::OMRuntimeShape::extendedShape(4, unextended_input_shape);
  const core::OMRuntimeShape output_shape =
    core::OMRuntimeShape::extendedShape(4, unextended_output_shape);

  const int output_batch = output_shape.dims(0);
  const int output_depth = output_shape.dims(3);

  const int input_height = input_shape.dims(1);
  const int input_width = input_shape.dims(2);

  for (int out_b = 0; out_b < output_batch; ++out_b)
  {
    for (int out_d = 0; out_d < output_depth; ++out_d)
    {
      float value = 0;
      for (int in_h = 0; in_h < input_height; ++in_h)
      {
        for (int in_w = 0; in_w < input_width; ++in_w)
        {
          value += input_data[offset(input_shape.dimsData(), out_b, in_h, in_w, out_d)];
        }
      }
      output_data[offset(output_shape.dimsData(), out_b, 0, 0, out_d)] =
        value / (input_width * input_height);
    }
  }
  return Ok;
}

} // namespace pal
} // namespace execute
} // namespace onert_micro

#endif // ONERT_MICRO_EXECUTE_PAL_MEAN_COMMON_H
