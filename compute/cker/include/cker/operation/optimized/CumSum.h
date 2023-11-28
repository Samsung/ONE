/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __NNFW_CKER_OPTIMIZED_CUMSUM_H__
#define __NNFW_CKER_OPTIMIZED_CUMSUM_H__

#include "cker/eigen/EigenSupport.h"
#include "cker/eigen/Utils.h"
#include "cker/Shape.h"

namespace nnfw
{
namespace cker
{
namespace optimized
{

template <typename T>
void CumsumImpl(const T *input_data, const Shape &shape, int axis, bool exclusive, bool reverse,
                T *output_data)
{
  Eigen::array<Eigen::DenseIndex, 3> dims = {1, 1, 1};

  for (int i = 0; i < axis; ++i)
  {
    dims[0] *= shape.Dims(i);
  }
  dims[1] = shape.Dims(axis);
  for (int i = axis + 1; i < shape.DimensionsCount(); ++i)
  {
    dims[2] *= shape.Dims(i);
  }

  typedef Eigen::TensorMap<Eigen::Tensor<const T, 3, Eigen::RowMajor, Eigen::DenseIndex>,
                           Eigen::Aligned>
    ConstTensor;
  typedef Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned>
    Tensor;
  ConstTensor input(input_data, dims);
  Tensor output(output_data, dims);

  if (reverse)
  {
    Eigen::array<bool, 3> reverse_idx = {false, true, false};
    output = input.reverse(reverse_idx).cumsum(1, exclusive).reverse(reverse_idx);
  }
  else
  {
    output = input.cumsum(1, exclusive);
  }
}

template <typename T>
void CumSum(const T *input_data, const Shape &shape, int axis, bool exclusive, bool reverse,
            T *output_data)
{
  assert(shape.DimensionsCount() >= 1);
  CumsumImpl<T>(input_data, shape, axis, exclusive, reverse, output_data);
}

} // namespace optimized
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_OPTIMIZED_CONV_H__
