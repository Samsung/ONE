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

#ifndef __NNFW_CKER_TRAIN_OPERATION_REGULARIZER_H__
#define __NNFW_CKER_TRAIN_OPERATION_REGULARIZER_H__

#include "cker/Shape.h"
#include "cker/eigen/Utils.h"

namespace nnfw
{
namespace cker
{
namespace train
{

template <typename T>
inline void L1(const Shape &incoming_shape, T *incoming_data, float lambda)
{
  double sum = 0.0f;
  const int size = incoming_shape.FlatSize();
  for (int i = 0; i < size; ++i)
  {
    sum += std::abs(incoming_data[i]);
  }

  auto reg = lambda * sum;
  for (int i = 0; i < size; ++i)
  {
    incoming_data[i] += reg;
  }
}

template <typename T>
inline void L2(const Shape &incoming_shape, T *incoming_data, float lambda)
{
  double sum = 0.0f;
  const int size = incoming_shape.FlatSize();
  for (int i = 0; i < size; ++i)
  {
    sum += incoming_data[i] * incoming_data[i];
  }

  auto reg = lambda * sum / 2.;
  for (int i = 0; i < size; ++i)
  {
    incoming_data[i] += reg;
  }
}

} // namespace train
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_TRAIN_OPERATION_REGULARIZER_H__
