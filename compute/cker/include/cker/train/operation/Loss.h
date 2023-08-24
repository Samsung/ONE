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

#include "cker/Shape.h"
#include "cker/eigen/Utils.h"

namespace nnfw
{
namespace cker
{
namespace train
{

template <typename T>
inline void MSE(const Shape &y_pred_shape, const T *y_pred_data, const Shape &y_true_shape,
                const T *y_true_data, const Shape &output_shape, T *output_data)
{
  // TODO Consider Reduction
  assert(output_shape == Shape{1});

  double squared_sum = 0.0f;
  const int size = MatchingFlatSize(y_pred_shape, y_true_shape);
  for (int i = 0; i < size; ++i)
  {
    squared_sum += std::pow(y_pred_data[i] - y_true_data[i], 2);
  }

  output_data[0] = static_cast<T>(squared_sum / size);
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

} // namespace train
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_TRAIN_OPERATION_LOSS_H__
