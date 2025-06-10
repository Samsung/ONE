/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __NNFW_CKER_EIGEN_UTILS_H__
#define __NNFW_CKER_EIGEN_UTILS_H__

#include <Eigen/Core>
#include <type_traits>
#include "cker/Shape.h"

namespace nnfw
{
namespace cker
{

// Make a local VectorMap typedef allowing to map a float array
// as a Eigen vector expression. The std::conditional here is to
// construct the suitable Eigen type for the constness of the
// data. Indeed, for const data, we need to produce
//    Eigen::Map<const Eigen::Matrix<float, ...>>
// and not the more straightforward
//    Eigen::Map<Eigen::Matrix<const float, ...>>
template <typename Scalar>
using VectorMap = typename std::conditional<
  std::is_const<Scalar>::value,
  Eigen::Map<const Eigen::Matrix<typename std::remove_const<Scalar>::type, Eigen::Dynamic, 1>>,
  Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>>::type;

template <typename Scalar> VectorMap<Scalar> MapAsVector(Scalar *data, const Shape &shape)
{
  const int size = shape.FlatSize();
  return VectorMap<Scalar>(data, size, 1);
}

// Make a local VectorMap typedef allowing to map a float array
// as a Eigen matrix expression. The same explanation as for VectorMap
// above also applies here.
template <typename Scalar>
using MatrixMap = typename std::conditional<
  std::is_const<Scalar>::value,
  Eigen::Map<
    const Eigen::Matrix<typename std::remove_const<Scalar>::type, Eigen::Dynamic, Eigen::Dynamic>>,
  Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>>::type;

template <typename Scalar>
MatrixMap<Scalar> MapAsMatrixWithLastDimAsRows(Scalar *data, const Shape &shape)
{
  const int dims_count = shape.DimensionsCount();
  const int rows = shape.Dims(dims_count - 1);
  const int cols = FlatSizeSkipDim(shape, dims_count - 1);
  return MatrixMap<Scalar>(data, rows, cols);
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_EIGEN_UTILS_H__
