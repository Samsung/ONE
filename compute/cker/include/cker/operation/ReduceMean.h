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

#ifndef __NNFW_CKER_REDUCEMEAN_H__
#define __NNFW_CKER_REDUCEMEAN_H__

#include "cker/Shape.h"
#include "cker/operation/Reduce.h"

namespace nnfw
{
namespace cker
{

float round_nearest(float value)
{
  if (value < 0)
  {
    return static_cast<float>(static_cast<int>(value - 0.5f));
  }
  else
  {
    return static_cast<float>(static_cast<int>(value + 0.5f));
  }
}
template <typename Out, typename In>
Out mean_reducer(const Out data1, const In data2, int normalizer)
{
  return data1 + static_cast<Out>(data2) / normalizer;
}

template <typename In> int sum_reducer(const int data1, const In data2)
{
  return data1 + static_cast<int>(data2);
}

template <typename In, typename Out>
inline bool ReduceMeanImpl(const In *input_data, const Shape &input_shape, const int *axis,
                           const int num_axis, int *input_iter,
                           Out reducer(const Out current, const In in, int normalizer),
                           Out *output_data)
{
  const auto input_dims = input_shape.DimsData();
  const auto input_num_dims = input_shape.DimensionsCount();
  int normalizer = 1;
  // Reset input iterator.
  for (int idx = 0; idx < input_num_dims; ++idx)
  {
    input_iter[idx] = 0;
  }
  // Compute number of output elements
  for (int idx = 0; idx < num_axis; ++idx)
  {
    normalizer *= input_dims[axis[idx]];
  }
  // Iterate through input_data.
  do
  {
    size_t input_offset = ReducedOutputOffset(input_num_dims, input_dims, input_iter, 0, nullptr);
    size_t output_offset =
        ReducedOutputOffset(input_num_dims, input_dims, input_iter, num_axis, axis);
    output_data[output_offset] =
        reducer(output_data[output_offset], input_data[input_offset], normalizer);
  } while (NextIndex(input_num_dims, input_dims, input_iter));
  return true;
}

template <typename In>
inline size_t ReduceSumQuantImpl(const In *input_data, const Shape &input_shape, const int *axis,
                                 const int num_axis, int *input_iter,
                                 int reducer(const int current, const In in), int *temp_sum)
{
  const auto input_dims = input_shape.DimsData();
  const auto input_num_dims = input_shape.DimensionsCount();
  size_t normalizer = 1;
  // Reset input iterator.
  for (int idx = 0; idx < input_num_dims; ++idx)
  {
    input_iter[idx] = 0;
  }
  // Compute number of output elements
  for (int idx = 0; idx < num_axis; ++idx)
  {
    normalizer *= input_dims[axis[idx]];
  }
  // Iterate through input_data.
  do
  {
    size_t input_offset = ReducedOutputOffset(input_num_dims, input_dims, input_iter, 0, nullptr);
    size_t output_offset =
        ReducedOutputOffset(input_num_dims, input_dims, input_iter, num_axis, axis);
    temp_sum[output_offset] = reducer(temp_sum[output_offset], input_data[input_offset]);
  } while (NextIndex(input_num_dims, input_dims, input_iter));
  return normalizer;
}

class ReduceMean : public Reduce
{
public:
  ReduceMean() : Reduce(){};

  template <typename T>
  int PrepareforReduce(const Shape &input_shape, const Shape &output_shape,
                       const std::vector<int> &axes, T *output_data, T init_value)
  {
    // Reset output data.
    if (!InitTensorDataForReduce(output_shape, init_value, output_data))
    {
      return -1;
    }
    const auto input_dims = input_shape.DimsData();
    const int num_dims = input_shape.DimensionsCount();
    int resolved_axis_size = 1;
    const auto num_axes = axes.size();

    for (size_t idx = 0; idx < num_axes; idx++)
    {
      int current = axes[idx] < 0 ? (axes[idx] + num_dims) : axes[idx];
      assert(current >= 0 && current < num_dims);
      resolved_axis_size *= input_dims[current];
    }

    prepare(num_dims, resolved_axis_size);

    // Resolve axis.
    int num_resolved_axis = 0;
    if (!ResolveAxis(input_shape.DimensionsCount(), axes, resolved_axis_data(), &num_resolved_axis))
    {
      return -1;
    }

    return num_resolved_axis;
  }

  // Computes the generic value (i.e., sum/max/min/prod) of elements across
  // dimensions given in axis. It needs to pass in init_value and reducer.
  template <typename In, typename Out>
  inline bool ReduceOp(const Shape &input_shape, const In *input_data, const Shape &output_shape,
                       Out *output_data, const std::vector<int> &axes, bool, Out init_value,
                       Out reducer(const Out current, const Out in, int normalizer))
  {
    int num_resolved_axis;
    num_resolved_axis = PrepareforReduce(input_shape, output_shape, axes, output_data, init_value);
    if (num_resolved_axis == -1)
    {
      return false;
    }
    return ReduceMeanImpl<In, Out>(input_data, input_shape, resolved_axis_data(), num_resolved_axis,
                                   temp_index_data(), reducer, output_data);
  }

  template <typename In, typename Out>
  inline bool ReduceOp(const Shape &input_shape, const In *input_data, float input_scale,
                       int32_t input_offset, const Shape &output_shape, Out *output_data,
                       float output_scale, int32_t output_offset, const std::vector<int> &axes,
                       bool, Out init_value, int reducer(const int current, const In in))
  {
    size_t num_outputs = 1;
    auto output_dims = output_shape.DimsData();

    for (size_t idx = 0; idx < static_cast<size_t>(output_shape.DimensionsCount()); idx++)
    {
      num_outputs *= output_dims[idx];
    }
    _temp_sum.resize(num_outputs, 0);
    int num_resolved_axis;
    num_resolved_axis = PrepareforReduce(input_shape, output_shape, axes, output_data, init_value);
    if (num_resolved_axis == -1)
    {
      return false;
    }

    size_t normalizer =
        ReduceSumQuantImpl<In>(input_data, input_shape, resolved_axis_data(), num_resolved_axis,
                               temp_index_data(), reducer, _temp_sum.data());
    if (num_outputs > 0)
    {
      float scale = input_scale / output_scale;
      float bias = -input_offset * scale;
      for (size_t idx = 0; idx < num_outputs; idx++)
      {
        float float_mean = static_cast<float>(_temp_sum[idx]) / normalizer;
        float result = std::min(round_nearest(float_mean * scale + bias + output_offset),
                                static_cast<float>(std::numeric_limits<Out>::max()));
        result = std::max(result, static_cast<float>(std::numeric_limits<Out>::min()));
        output_data[idx] = static_cast<Out>(result);
      }
    }
    return false;
  }

private:
  std::vector<int> _temp_sum;
};

template <typename In, typename Out>
void Mean(const Shape &input_shape, const In *input_data, const Shape &output_shape,
          Out *output_data, const std::vector<int> &axes)
{
  UNUSED_RELEASE(output_shape);
  assert(input_shape.DimensionsCount() > 0);
  ReduceMean m_obj;
  m_obj.ReduceOp<In, Out>(input_shape, input_data, output_shape, output_data, axes, true, (Out)0,
                          mean_reducer);
}

template <typename In, typename Out>
void MeanQ8Asymm(const Shape &input_shape, const In *input_data, float input_scale,
                 int32_t input_offset, const Shape &output_shape, Out *output_data,
                 float output_scale, int32_t output_offset, const std::vector<int> &axes)
{
  UNUSED_RELEASE(output_shape);
  assert(input_shape.DimensionsCount() > 0);
  ReduceMean m_obj;
  m_obj.ReduceOp<In, Out>(input_shape, input_data, input_scale, input_offset, output_shape,
                          output_data, output_scale, output_offset, axes, true, (Out)0,
                          sum_reducer);
}

template <typename In, typename Out>
void MeanAxis1And2(const Shape &input_shape, const In *input_data, const Shape &output_shape,
                   Out *output_data)
{
  UNUSED_RELEASE(output_shape);
  assert(input_shape.DimensionsCount() == 4);
  assert(output_shape.DimensionsCount() == 4);

  const int output_batch = output_shape.Dims(0);
  const int output_depth = output_shape.Dims(3);

  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);

  for (int out_b = 0; out_b < output_batch; ++out_b)
  {
    for (int out_d = 0; out_d < output_depth; ++out_d)
    {
      float value = 0;
      for (int in_h = 0; in_h < input_height; ++in_h)
      {
        for (int in_w = 0; in_w < input_width; ++in_w)
        {
          value += input_data[Offset(input_shape, out_b, in_h, in_w, out_d)];
        }
      }
      output_data[Offset(output_shape, out_b, 0, 0, out_d)] = value / (input_width * input_height);
    }
  }
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_REDUCEMEAN_H__
