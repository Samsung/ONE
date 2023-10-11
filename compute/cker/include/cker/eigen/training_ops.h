/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __NNFW_CKER_EIGEN_TRAINING_OPS_H__
#define __NNFW_CKER_EIGEN_TRAINING_OPS_H__

// From tensorflow/core/kernels/training_ops.cc
#define EIGEN_USE_THREADS

#include "unsupported/Eigen/CXX11/Tensor"
#include "cker/operation/Helper/Tensor.h"

// From tensorflow/core/kernels/training_ops.h
namespace nnfw
{
namespace cker
{
namespace training_ops
{
namespace functor
{

template <typename Device, typename T> struct ApplyAdam
{
  void operator()(const Device &d, typename TTypes<T>::Flat var, typename TTypes<T>::Flat m,
                  typename TTypes<T>::Flat v, typename TTypes<T>::ConstScalar beta1_power,
                  typename TTypes<T>::ConstScalar beta2_power, typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar beta1, typename TTypes<T>::ConstScalar beta2,
                  typename TTypes<T>::ConstScalar epsilon, typename TTypes<T>::ConstFlat grad,
                  bool use_nesterov);
};

} // namespace functor
} // namespace training_ops
} // namespace cker
} // namespace nnfw

// From tensorflow/core/kernels/training_ops.cc
namespace nnfw
{
namespace cker
{
namespace training_ops
{

// Enable CPUDevice only for training_ops
using CPUDevice = Eigen::ThreadPoolDevice;
using Index = Eigen::Index;

namespace functor
{

template <typename Device, typename T> struct ApplyAdamNonCuda
{
  void operator()(const Device &d, typename TTypes<T>::Flat var, typename TTypes<T>::Flat m,
                  typename TTypes<T>::Flat v, typename TTypes<T>::ConstScalar beta1_power,
                  typename TTypes<T>::ConstScalar beta2_power, typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar beta1, typename TTypes<T>::ConstScalar beta2,
                  typename TTypes<T>::ConstScalar epsilon, typename TTypes<T>::ConstFlat grad,
                  bool use_nesterov)
  {
    // Get params length and check if they can be vectorized by packet size.
    Index length = var.size();
    Index packet_size = Eigen::internal::packet_traits<T>::size;
    if (length % packet_size == 0)
    {
      length = length / packet_size;
    }
    else
    {
      packet_size = 1;
    }

    T *var_ptr = var.data();
    T *m_ptr = m.data();
    T *v_ptr = v.data();
    const T *g_ptr = grad.data();
    const T alpha = lr() * Eigen::numext::sqrt(T(1) - beta2_power()) / (T(1) - beta1_power());
    // beta1 == μ
    // beta2 == ν
    // v     == n
    // var   == θ

    auto shard = [var_ptr, m_ptr, v_ptr, g_ptr, alpha, beta1, beta2, epsilon, use_nesterov,
                  packet_size](int begin, int end) {
      int t_size = (end - begin) * packet_size;
      begin = begin * packet_size;
      auto var = typename TTypes<T>::UnalignedTensor(var_ptr + begin, t_size);
      auto m = typename TTypes<T>::UnalignedTensor(m_ptr + begin, t_size);
      auto v = typename TTypes<T>::UnalignedTensor(v_ptr + begin, t_size);
      auto g = typename TTypes<T>::UnalignedConstTensor(g_ptr + begin, t_size);

      if (use_nesterov)
      {
        m += (g - m) * (T(1) - beta1());
        v += (g.square() - v) * (T(1) - beta2());
        var -= ((g * (T(1) - beta1()) + beta1() * m) * alpha) / (v.sqrt() + epsilon());
      }
      else
      {
        m += (g - m) * (T(1) - beta1());
        v += (g.square() - v) * (T(1) - beta2());
        var -= (m * alpha) / (v.sqrt() + epsilon());
      }
    };

    // Input data: var, v, m, grad.
    // Output data: var, v, m.
    const int input_bytes = length * packet_size * sizeof(T) * 4;
    const int output_bytes = length * packet_size * sizeof(T) * 3;
    const int compute_cycles =
      // Consider Sub as Add
      (Eigen::TensorOpCost::AddCost<int>() * 5 + Eigen::TensorOpCost::MulCost<int>() * 2 +
       Eigen::TensorOpCost::AddCost<T>() * 10 + Eigen::TensorOpCost::MulCost<T>() * 6 +
       Eigen::TensorOpCost::DivCost<T>()) *
      length;
    const Eigen::TensorOpCost cost(input_bytes, output_bytes, compute_cycles);

    // Eigen device must update 3 variables with 3 different expressions,
    // which is bad for cache locality on CPU. Here use ParallelFor instead of
    // "regular" tensor expressions to get better performance.
    d.parallelFor(length, cost, shard);
  }
};

template <typename T> struct ApplyAdam<CPUDevice, T> : ApplyAdamNonCuda<CPUDevice, T>
{
};

} // namespace functor
} // namespace training_ops
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_EIGEN_TRAINING_OPS_H__
