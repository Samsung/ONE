/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __NNFW_CKER_LOGISTIC_H__
#define __NNFW_CKER_LOGISTIC_H__

#include "cker/Shape.h"
#include "cker/eigen/Utils.h"

#include <cmath>
#include <Eigen/Core>

namespace nnfw
{
namespace cker
{

inline void Logistic(const Shape &input_shape, const float *input_data, const Shape &output_shape,
                     float *output_data)
{
  auto input_map = MapAsVector(input_data, input_shape);
  auto output_map = MapAsVector(output_data, output_shape);

  output_map.array() = input_map.array().unaryExpr(Eigen::internal::scalar_logistic_op<float>());
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_LOGISTIC_H__
