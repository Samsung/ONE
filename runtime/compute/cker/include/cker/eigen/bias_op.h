/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __NNFW_CKER_EIGEN_BIAS_OP_H__
#define __NNFW_CKER_EIGEN_BIAS_OP_H__

// From tensorflow/core/kernels/bias_op.cc
#define EIGEN_USE_THREADS

#include "unsupported/Eigen/CXX11/Tensor"
#include "cker/operation/Helper/Tensor.h"

// From tensorflow/core/kernels/bias_op.h
namespace nnfw
{
namespace cker
{
namespace bias_op
{

namespace functor
{

namespace internal
{

template <typename Device> struct MaybeWith32BitIndexingImpl
{
  template <typename Func, typename... Args> void operator()(Func func, Args &&...args) const
  {
    func(std::forward<Args>(args)...);
  }
};

} // namespace internal

template <typename Device, typename Func, typename... Args>
void MaybeWith32BitIndexing(Func func, Args &&...args)
{
  return internal::MaybeWith32BitIndexingImpl<Device>()(func, std::forward<Args>(args)...);
}

// Functor used by BiasOp to do the computations.
// NOTE Apply activation to Bias
template <typename Device, typename T> struct Bias
{
  // Add "bias" to "input", repeating "bias".
  void operator()(const Device &d, typename TTypes<T>::ConstFlat input,
                  typename TTypes<T>::ConstVec bias, typename TTypes<T>::Flat output,
                  T activation_min, T activation_max)
  {
    const Eigen::Index rest_size = input.size() / bias.dimension(0);
    Eigen::DSizes<Eigen::Index, 1> bcast(rest_size);
    MaybeWith32BitIndexing<Device>(
      [&](auto input32, auto bias32, typename TTypes<T>::Flat output32, const auto &bcast32,
          T activation_min, T activation_max) {
        output32.device(d) =
          (input32 + bias32.broadcast(bcast32))
            .template cwiseMax<Eigen::PropagateNaN>(static_cast<T>(activation_min))
            .template cwiseMin<Eigen::PropagateNaN>(static_cast<T>(activation_max));
      },
      input, bias, output, bcast, activation_min, activation_max);
  }
};

} // namespace functor
} // namespace bias_op
} // namespace cker
} // namespace nnfw

// From tensorflow/core/kernels/bias_op.cc
namespace nnfw
{
namespace cker
{
namespace bias_op
{

// Enable CPUDevice only for depthwise_conv_op
using Device = Eigen::ThreadPoolDevice;

template <typename T>
void biasHelper(const Shape &bias_shape, const T *bias_data, const Shape &input_shape,
                T *input_data, T activation_min, T activation_max)
{
  [[maybe_unused]] int channel_dim = input_shape.DimensionsCount() - 1;

  assert(input_shape.Dims(channel_dim) == bias_shape.Dims(0));
  assert(input_data);
  assert(bias_data);

  Tensor bias{bias_shape, const_cast<T *>(bias_data)};
  Tensor input{input_shape, input_data};

  functor::Bias<Device, T> functor;
  const Eigen::ThreadPoolDevice &d = *eigen_support::GetThreadPoolDevice();
  functor(d, static_cast<const Tensor &>(input).flat<T>(),
          static_cast<const Tensor &>(bias).flat<T>(), input.flat<T>(), activation_min,
          activation_max);
}

} // namespace bias_op
} // namespace cker
} // namespace nnfw
#endif // __NNFW_CKER_EIGEN_BIAS_OP_H__
