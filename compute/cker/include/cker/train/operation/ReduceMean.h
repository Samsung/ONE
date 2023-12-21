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

#ifndef __NNFW_CKER_TRAIN_OPERATION_REDUCEMEAN_H__
#define __NNFW_CKER_TRAIN_OPERATION_REDUCEMEAN_H__

#include "cker/Shape.h"
#include "cker/eigen/Utils.h"
#include "cker/operation/BroadcastTo.h"

namespace nnfw
{
namespace cker
{
namespace train
{

template <typename T>
void MeanGrad(const Shape &incoming_shape, const T *incoming_data, const Shape &grad_shape,
              T *grad_data)
{
  BroadcastTo(incoming_shape, const_cast<T*>(incoming_data), grad_shape, grad_data);
  const auto incoming = MapAsMatrixWithLastDimAsRows(incoming_data, incoming_shape);
  auto grad = MapAsMatrixWithLastDimAsRows(grad_data, grad_shape);
  grad /= (grad.size() / incoming.size());
}

} // namespace train
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_TRAIN_OPERATION_REDUCEMEAN_H__
