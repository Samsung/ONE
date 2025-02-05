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

#ifndef __NNFW_CKER_TRAIN_OPERATION_FULLY_CONNECTED_H__
#define __NNFW_CKER_TRAIN_OPERATION_FULLY_CONNECTED_H__

#include "cker/eigen/Utils.h"
#include "cker/Shape.h"

namespace nnfw
{
namespace cker
{
namespace train
{

template <typename T>
inline void FullyConnectedBiasGrad(const Shape &incomming_shape, const T *incomming_data,
                                   const Shape &grad_shape, T *grad_data)
{
  const auto bias_size = grad_shape.FlatSize();
  if (bias_size != incomming_shape.Dims(incomming_shape.DimensionsCount() - 1) ||
      bias_size != grad_shape.Dims(0))
    throw std::runtime_error("cker::FullyConnectedBiasGrad: Unmatched shape");

  const auto in_mat = MapAsMatrixWithLastDimAsRows(incomming_data, incomming_shape);
  auto grad_mat = MapAsMatrixWithLastDimAsRows(grad_data, grad_shape);

  grad_mat = in_mat.rowwise().sum();
}

} // namespace train
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_FULLY_CONNECTED_H__
