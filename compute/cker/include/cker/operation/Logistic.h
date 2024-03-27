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

/**
 * @brief Internal scalar_logistic_op operation struct
 *
 * @note  Recent Eigen3 scalar_logistic_op return invalid value on ARM32 if
 *        input value is float type 88 (expected: 1, actual: 0)
 *        As a workaround, we use old version scalar_logistic_op internal struct
 *        TODO Remove this workaround
 */
template <typename T> struct scalar_logistic_op
{
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T &x) const
  {
    const T one = T(1);
    return one / (one + Eigen::numext::exp(-x));
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet &x) const
  {
    const Packet one = Eigen::internal::pset1<Packet>(T(1));
    return pdiv(one, padd(one, pexp(pnegate(x))));
  }
};

inline void Logistic(const Shape &input_shape, const float *input_data, const Shape &output_shape,
                     float *output_data)
{
  auto input_map = MapAsVector(input_data, input_shape);
  auto output_map = MapAsVector(output_data, output_shape);

  // Use old version scalar_logistic_op
  output_map.array() = input_map.array().unaryExpr(nnfw::cker::scalar_logistic_op<float>());
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_LOGISTIC_H__
