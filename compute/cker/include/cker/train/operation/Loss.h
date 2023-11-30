/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __NNFW_CKER_TRAIN_OPERATION_LOSS_H__
#define __NNFW_CKER_TRAIN_OPERATION_LOSS_H__

#include <numeric>

#include "cker/Shape.h"
#include "cker/eigen/Utils.h"

namespace nnfw
{
namespace cker
{
namespace train
{

template <typename T> inline T square(T value) { return value * value; }
template <typename T> inline T log_threshold() { return static_cast<T>(1e-20); }

template <typename T>
inline void MSE(const Shape &y_pred_shape, const T *y_pred_data, const Shape &y_true_shape,
                const T *y_true_data, const Shape &output_shape, T *output_data)
{
  if (output_shape.DimensionsCount() != 1)
    throw std::runtime_error("cker::MSE: output dimension count should be 1");
  if (output_shape.Dims(0) != y_pred_shape.Dims(0))
    throw std::runtime_error("cker::MSE: output and y_pred do not have the same batch");
  if (y_pred_shape != y_true_shape)
    throw std::runtime_error("cker::MSE: y_pred_shape != y_true_shape");

  const auto batch = y_pred_shape.Dims(0);
  const auto size = FlatSizeSkipDim(y_pred_shape, 0);

  for (int b = 0; b < batch; ++b)
  {
    float sum = 0.f;
    for (int i = 0; i < size; ++i)
    {
      sum += square(y_pred_data[b * size + i] - y_true_data[b * size + i]);
    }
    output_data[b] = static_cast<T>(sum / size);
  }
}

template <typename T>
inline void MSEGrad(const Shape &y_pred_shape, const T *y_pred_data, const Shape &y_true_shape,
                    const T *y_true_data, const Shape &grad_shape, T *grad_data)
{
  if (y_pred_shape != y_true_shape)
    throw std::runtime_error("cker::MSEGrad: y_pred_shape != y_true_shape");
  if (y_pred_shape != grad_shape)
    throw std::runtime_error("cker::MSEGrad: y_pred_shape != grad_shape");

  const int size = grad_shape.FlatSize();
  for (int i = 0; i < size; ++i)
  {
    grad_data[i] = static_cast<T>(-2 * (y_true_data[i] - y_pred_data[i]) / size);
  }
}

template <typename T>
inline void CategoricalCrossEntropy(const Shape &y_pred_shape, const T *y_pred_data,
                                    const Shape &y_true_shape, const T *y_true_data,
                                    const Shape &output_shape, T *output_data)
{
  if (output_shape.DimensionsCount() != 1)
    throw std::runtime_error("cker::CategoricalCrossEntropy: output dimension count should be 1");
  if (y_pred_shape != y_true_shape)
    throw std::runtime_error(
      "cker::CategoricalCrossEntropy: y_pred and y_true do not have the same shape");
  if (output_shape.Dims(0) != y_pred_shape.Dims(0))
    throw std::runtime_error(
      "cker::CategoricalCrossEntropy: output and y_pred do not have the same batch");

  const auto y_pred = MapAsMatrixWithLastDimAsRows(y_pred_data, y_pred_shape);
  const auto y_true = MapAsMatrixWithLastDimAsRows(y_true_data, y_true_shape);
  auto output = MapAsVector(output_data, output_shape);

  output = -(y_true.array() * y_pred.array().cwiseMax(log_threshold<T>()).log()).colwise().sum();
}

template <typename T>
inline void CategoricalCrossEntropyGrad(const Shape &y_pred_shape, const T *y_pred_data,
                                        const Shape &y_true_shape, const T *y_true_data,
                                        const Shape &grad_shape, T *grad_data)
{
  if (y_pred_shape != y_true_shape)
    throw std::runtime_error(
      "cker::CategoricalCrossEntropyGrad: y_pred and y_true do not have the same shape");
  if (y_pred_shape != grad_shape)
    throw std::runtime_error(
      "cker::CategoricalCrossEntropyGrad: y_pred and grad do not have the same shape");

  const auto y_pred = MapAsMatrixWithLastDimAsRows(y_pred_data, y_pred_shape);
  const auto y_true = MapAsMatrixWithLastDimAsRows(y_true_data, y_true_shape);
  auto grad = MapAsMatrixWithLastDimAsRows(grad_data, grad_shape);

  grad = -(y_true.array() / y_pred.array().cwiseMax(log_threshold<T>()));
}

} // namespace train
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_TRAIN_OPERATION_LOSS_H__
