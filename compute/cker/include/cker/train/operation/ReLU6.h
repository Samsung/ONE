/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __NNFW_CKER_TRAIN_OPERATION_RELU6_H__
#define __NNFW_CKER_TRAIN_OPERATION_RELU6_H__

#include "cker/Shape.h"
#include "cker/eigen/Utils.h"
#include <Eigen/Core>

namespace nnfw
{
namespace cker
{
namespace train
{

inline void ReLU6Grad(const Shape &output_shape, const float *output_data,
                      const Shape &incoming_shape, const float *incoming_data,
                      const Shape &grad_shape, float *grad_data)
{
  const auto output_map = MapAsVector(output_data, output_shape);
  const auto incoming_map = MapAsVector(incoming_data, incoming_shape);
  auto grad_map = MapAsVector(grad_data, grad_shape);

  if (output_shape == incoming_shape && output_shape == grad_shape)
    grad_map.array() =
      incoming_map.array() *
      (0.0f < output_map.array() && output_map.array() < 6.0f).template cast<float>();
  else
    throw std::runtime_error("cker::ReLUGrad: Unsupported shape");
}

} // namespace train
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_TRAIN_OPERATION_RELU6_H__
