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

template <typename T> T mean_reducer(T data1, T data2, int normalizer)
{
  return data1 + data2 * 1.0 / normalizer;
}

template <typename In, typename Out>
inline bool ReduceMeanImpl(const In *input_data, const Shape &input_shape, const Shape &,
                           const int *axis, const int num_axis, int *input_iter,
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

class ReduceMean : public Reduce
{
public:
  ReduceMean() : Reduce(){};

  // Computes the generic value (i.e., sum/max/min/prod) of elements across
  // dimensions given in axis. It needs to pass in init_value and reducer.
  template <typename T>
  inline bool ReduceOp(const Shape &input_shape, const T *input_data, const Shape &output_shape,
                       T *output_data, const std::vector<int> &axes, bool, T init_value,
                       T reducer(const T current, const T in, int normalizer))
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

    return ReduceMeanImpl<T, T>(input_data, input_shape, output_shape, resolved_axis_data(),
                                num_resolved_axis, temp_index_data(), reducer, output_data);
  }
};

template <typename T1, typename T2>
void Mean(const Shape &input_shape, const T1 *input_data, const Shape &output_shape,
          T2 *output_data, std::vector<int> &axes)
{
  UNUSED_RELEASE(output_shape);
  assert(input_shape.DimensionsCount() > 0);
  ReduceMean m_obj;
  m_obj.ReduceOp(input_shape, input_data, output_shape, output_data, axes, true, (T2)0,
                 mean_reducer);
}
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_REDUCEMEAN_H__
