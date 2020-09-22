/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __NNFW_CKER_REDUCE_H__
#define __NNFW_CKER_REDUCE_H__

#include "cker/Shape.h"
#include "cker/Types.h"
#include "cker/Utils.h"

namespace nnfw
{
namespace cker
{

// A generic reduce method that can be used for reduce_sum, reduce_mean, etc.
// This method iterates through input data and reduce elements along the
// dimensions given in axis.
template <typename In, typename Out>
inline bool ReduceImpl(const In *input_data, const Shape &input_shape, const Shape &,
                       const int *axis, const int num_axis, int *input_iter,
                       Out reducer(const Out current, const In in), Out *output_data)
{
  const auto input_dims = input_shape.DimsData();
  const auto input_num_dims = input_shape.DimensionsCount();

  // Reset input iterator.
  if (num_axis == 1 && (axis[0] == -1 || axis[0] == input_num_dims - 1))
  {
    int input_size = 1;
    int reduce_size = 0;
    for (int idx = 0; idx < input_num_dims - 1; idx++)
    {
      input_size *= input_dims[idx];
    }
    reduce_size = input_dims[input_num_dims - 1];
    for (int idx = 0; idx < input_size; idx++)
    {
      for (int r_idx = 0; r_idx < reduce_size; r_idx++)
      {
        if (r_idx == 0)
        {
          output_data[idx] = input_data[idx * reduce_size];
        }
        else
        {
          output_data[idx] = reducer(output_data[idx], input_data[idx * reduce_size + r_idx]);
        }
      }
    }
    return true;
  }

  for (int idx = 0; idx < input_num_dims; ++idx)
  {
    input_iter[idx] = 0;
  }
  // Iterate through input_data.
  do
  {
    size_t input_offset = ReducedOutputOffset(input_num_dims, input_dims, input_iter, 0, nullptr);
    size_t output_offset =
        ReducedOutputOffset(input_num_dims, input_dims, input_iter, num_axis, axis);
    output_data[output_offset] = reducer(output_data[output_offset], input_data[input_offset]);
  } while (NextIndex(input_num_dims, input_dims, input_iter));
  return true;
}

// This method parses the input 'axis' to remove duplicates and handle negative
// values, and returns a valid 'out_axis'
inline bool ResolveAxis(const int num_dims, const std::vector<int> &axes, int *out_axis,
                        int *out_num_axis)
{
  auto num_axis = axes.size();
  auto axis = axes.data();

  *out_num_axis = 0; // Just in case.
  // Short-circuit axis resolution for scalars; the axis will go unused.
  if (num_dims == 0)
  {
    return true;
  }
  // o(n^2) is fine since out_num_axis should be really small, mostly <= 4
  for (size_t idx = 0; idx < num_axis; ++idx)
  {
    // Handle negative index. A positive index 'p_idx' can be represented as a
    // negative index 'n_idx' as: n_idx = p_idx-num_dims
    // eg: For num_dims=3, [0, 1, 2] is the same as [-3, -2, -1]  */
    int current = axis[idx] < 0 ? (axis[idx] + num_dims) : axis[idx];
    assert(current >= 0 && current < num_dims);
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

template <typename T>
inline bool InitTensorDataForReduce(const Shape &shape, const T init_value, T *data)
{
  const auto dims = shape.DimsData();
  const auto num_dims = shape.DimensionsCount();
  size_t num_elements = 1;
  for (int idx = 0; idx < num_dims; ++idx)
  {
    size_t current = static_cast<size_t>(dims[idx]);
    // Overflow prevention.
    if (num_elements > std::numeric_limits<size_t>::max() / current)
    {
      return false;
    }
    num_elements *= current;
  }
  for (size_t idx = 0; idx < num_elements; ++idx)
  {
    data[idx] = init_value;
  }
  return true;
}

class Reduce
{
public:
  Reduce() : _temp_index(), _resolved_axis(), _prepared(false) {}

  void prepare(size_t temp_index_size, size_t resolved_axis_size)
  {
    if (_prepared)
      return;

    // prepare space for temp_index and resolved_axis
    if (temp_index_size > kMaxSmallSize)
      _temp_index.resize(temp_index_size);
    if (resolved_axis_size > kMaxSmallSize)
      _resolved_axis.resize(resolved_axis_size);
    _prepared = true;
  }

  // Computes the generic value (i.e., sum/max/min/prod) of elements across
  // dimensions given in axis. It needs to pass in init_value and reducer.
  template <typename T>
  inline bool ReduceGeneric(const Shape &input_shape, const T *input_data,
                            const Shape &output_shape, T *output_data, const std::vector<int> &axes,
                            bool, T init_value, T reducer(const T current, const T in))
  {
    // Reset output data.
    if (!InitTensorDataForReduce(output_shape, init_value, output_data))
    {
      return false;
    }

    // Resolve axis.
    int num_resolved_axis = 0;
    if (!ResolveAxis(input_shape.DimensionsCount(), axes, resolved_axis_data(), &num_resolved_axis))
    {
      return false;
    }

    return ReduceImpl<T, T>(input_data, input_shape, output_shape, resolved_axis_data(),
                            num_resolved_axis, temp_index_data(), reducer, output_data);
  }

  // Computes the mean of elements across dimensions given in axis.
  // It does so in two stages, first calculates the sum of elements along the axis
  // then divides it by the number of element in axis for quantized values.
  template <typename T, typename U>
  inline bool QuantizedMeanOrSum(const T *input_data, int32_t input_zero_point, float input_scale,
                                 const Shape &input_shape, T *output_data,
                                 int32_t output_zero_point, float output_scale,
                                 const Shape &output_shape, const std::vector<int> &axes,
                                 bool /*keep_dims*/, U *temp_sum, bool compute_sum,
                                 U reducer(const U current, const T in))
  {
    // Reset output data.
    size_t num_outputs = 1;
    for (int idx = 0; idx < output_shape.DimensionsCount(); ++idx)
    {
      size_t current = static_cast<size_t>(output_shape.Dims(idx));
      // Overflow prevention.
      if (num_outputs > std::numeric_limits<size_t>::max() / current)
      {
        return false;
      }
      num_outputs *= current;
    }
    for (size_t idx = 0; idx < num_outputs; ++idx)
    {
      output_data[idx] = T();
      temp_sum[idx] = U();
    }

    // Resolve axis.
    int num_resolved_axis = 0;
    if (!ResolveAxis(input_shape.DimensionsCount(), axes, resolved_axis_data(), &num_resolved_axis))
    {
      return false;
    }

    if (!ReduceImpl<T, U>(input_data, input_shape, output_shape, resolved_axis_data(),
                          num_resolved_axis, temp_index_data(), reducer, temp_sum))
    {
      return false;
    }

    // Calculate mean by dividing output_data by num of aggregated element.
    U num_elements_in_axis = 1;
    for (int idx = 0; idx < num_resolved_axis; ++idx)
    {
      size_t current = static_cast<size_t>(input_shape.Dims(resolved_axis_data()[idx]));
      // Overflow prevention.
      if (current > static_cast<size_t>(std::numeric_limits<U>::max() / num_elements_in_axis))
      {
        return false;
      }
      num_elements_in_axis *= current;
    }

    if (num_elements_in_axis > 0)
    {
      const float scale = input_scale / output_scale;
      if (compute_sum)
      {
        // TODO(b/116341117): Eliminate float and do this completely in 8bit.
        const float bias = -input_zero_point * scale * num_elements_in_axis + 0.5f;
        for (size_t idx = 0; idx < num_outputs; ++idx)
        {
          const U value =
              static_cast<U>(std::round(temp_sum[idx] * scale + bias)) + output_zero_point;
          output_data[idx] = static_cast<T>(value);
        }
      }
      else
      {
        const float bias = -input_zero_point * scale + 0.5f;
        for (size_t idx = 0; idx < num_outputs; ++idx)
        {
          float float_mean =
              static_cast<float>(temp_sum[idx]) / static_cast<float>(num_elements_in_axis);
          float result = std::min(std::round(float_mean * scale + bias) + output_zero_point,
                                  static_cast<float>(std::numeric_limits<T>::max()));
          result = std::max(result, static_cast<float>(std::numeric_limits<T>::min()));
          output_data[idx] = static_cast<T>(result);
        }
      }
    }
    return true;
  }

  inline int32_t *resolved_axis_data(void)
  {
    return _resolved_axis.size() ? _resolved_axis.data() : _resolved_axis_small;
  }
  inline int32_t *temp_index_data(void)
  {
    return _temp_index.size() ? _temp_index.data() : _temp_index_small;
  }

private:
  std::vector<int> _temp_index;
  std::vector<int> _resolved_axis;
  bool _prepared;
  static constexpr int kMaxSmallSize = 4;
  int _temp_index_small[kMaxSmallSize];
  int _resolved_axis_small[kMaxSmallSize];
};

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_REDUCE_H__
