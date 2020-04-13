/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright (C) 2017 The Android Open Source Project
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

#ifndef __MATRIX_H__
#define __MATRIX_H__

#include "Dims.h"
#include "Eigen/Core"

// From optimized_ops.h in TensorFlow Lite
//
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

template <typename Scalar, int N> VectorMap<Scalar> MapAsVector(Scalar *data, const Dims<N> &dims)
{
  const int size = RequiredBufferSizeForDims(dims);
  return VectorMap<Scalar>(data, size, 1);
}

// From optimized_ops.h in TensorFlow Lite
//
// Make a local VectorMap typedef allowing to map a float array
// as a Eigen matrix expression. The same explanation as for VectorMap
// above also applies here.
template <typename Scalar>
using MatrixMap = typename std::conditional<
    std::is_const<Scalar>::value,
    Eigen::Map<const Eigen::Matrix<typename std::remove_const<Scalar>::type, Eigen::Dynamic,
                                   Eigen::Dynamic>>,
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>>::type;

// From optimized_ops.h in TensorFlow Lite
template <typename Scalar, int N>
MatrixMap<Scalar> MapAsMatrixWithFirstDimAsRows(Scalar *data, const Dims<N> &dims)
{
  const int rows = dims.sizes[0];
  int cols = 1;
  for (int d = 1; d < N; d++)
  {
    cols *= dims.sizes[d];
  }
  return MatrixMap<Scalar>(data, rows, cols);
}

// From optimized_ops.h in TensorFlow Lite
template <typename Scalar, int N>
MatrixMap<Scalar> MapAsMatrixWithLastDimAsCols(Scalar *data, const Dims<N> &dims)
{
  const int cols = dims.sizes[N - 1];
  int rows = 1;
  for (int d = 0; d < N - 1; d++)
  {
    rows *= dims.sizes[d];
  }
  return MatrixMap<Scalar>(data, rows, cols);
}

// From optimized_ops.h in TensorFlow Lite
template <typename Scalar>
using ArrayMap = typename std::conditional<
    std::is_const<Scalar>::value,
    Eigen::Map<const Eigen::Array<typename std::remove_const<Scalar>::type, Eigen::Dynamic,
                                  Eigen::Dynamic>>,
    Eigen::Map<Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic>>>::type;

// From optimized_ops.h in TensorFlow Lite
template <typename Scalar, int N>
ArrayMap<Scalar> MapAsArrayWithFirstDimAsRows(Scalar *data, const Dims<N> &dims)
{
  const int rows = dims.sizes[0];
  int cols = 1;
  for (int d = 1; d < N; d++)
  {
    cols *= dims.sizes[d];
  }
  return ArrayMap<Scalar>(data, rows, cols);
}

// From optimized_ops.h in TensorFlow Lite
//
// TODO(b/62193649): this function is only needed as long
// as we have the --variable_batch hack.
template <typename Scalar, int N>
MatrixMap<Scalar> MapAsMatrixWithGivenNumberOfRows(Scalar *data, const Dims<N> &dims, int rows)
{
  int cols = 1;
  bool matched_rows = false;
  for (int d = 0; d < N; d++)
  {
    cols *= dims.sizes[d];
    if (cols == rows)
    {
      matched_rows = true;
      cols = 1;
    }
  }
  DCHECK(matched_rows);
  return MatrixMap<Scalar>(data, rows, cols);
}

#endif // __MATRIX_H__
