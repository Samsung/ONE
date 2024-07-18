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
template <typename Device, typename T> struct Bias
{
  // Add "bias" to "input", repeating "bias".
  void operator()(const Device &d, typename TTypes<T>::ConstFlat input,
                  typename TTypes<T>::ConstVec bias, typename TTypes<T>::Flat output)
  {
    const Eigen::Index rest_size = input.size() / bias.dimension(0);
    Eigen::DSizes<Eigen::Index, 1> bcast(rest_size);
    MaybeWith32BitIndexing<Device>(
      [&](auto input32, auto bias32, auto output32, const auto &bcast32) {
        output32.device(d) = input32 + bias32.broadcast(bcast32);
      },
      input, bias, output, bcast);
  }
};

// Functor used by BiasOp to do the computations.
template <typename Device, typename T> struct BiasActivation
{
  // Add "bias" to "input", repeating "bias".
  void operator()(const Device &d, typename TTypes<T>::ConstFlat input,
                  typename TTypes<T>::ConstVec bias, typename TTypes<T>::Flat output,
                  int activation_min, int activation_max)
  {
    const Eigen::Index rest_size = input.size() / bias.dimension(0);
    Eigen::DSizes<Eigen::Index, 1> bcast(rest_size);
    MaybeWith32BitIndexing<Device>(
      [&](auto input32, auto bias32, typename TTypes<T>::Flat output32, const auto &bcast32,
          int activation_min, int activation_max) {
        output32.device(d) =
          (input32 + bias32.broadcast(bcast32))
            .template cwiseMax<Eigen::PropagateNaN>(static_cast<T>(activation_min))
            .template cwiseMin<Eigen::PropagateNaN>(static_cast<T>(activation_max));
        // if (temp < activation_min)
        //   temp = activation_min;
        // if (temp > activation_max)
        //     temp = activation_max;
        // output32.device(d) = temp;
        // output32.device(d) = input32 + bias32.broadcast(bcast32);
        // output32.device(d) = output32.cwiseMin(activation_min).cwiseMax(activation_max);
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

void biasHelper(const Shape &input_shape, const float *input_data, const Shape &bias_shape,
                const float *bias_data, const Shape &output_shape, float *output_data)
{
  assert(input_shape.Dims(3) == bias_shape.Dims(0));
  assert(input_data);
  assert(bias_data);

  Tensor input{input_shape, const_cast<float *>(input_data)};
  Tensor bias{bias_shape, const_cast<float *>(bias_data)};
  Tensor output{output_shape, output_data};

  functor::Bias<Device, float> functor;
  const Eigen::ThreadPoolDevice &d = *eigen_support::GetThreadPoolDevice();
  functor(d, static_cast<const Tensor &>(input).flat<float>(),
          static_cast<const Tensor &>(bias).flat<float>(), output.flat<float>());
}

void biasActivationHelper(const Shape &input_shape, const float *input_data,
                          const Shape &bias_shape, const float *bias_data,
                          const Shape &output_shape, float *output_data, int activation_min,
                          int activation_max)
{
  assert(input_shape.Dims(3) == bias_shape.Dims(0));
  assert(input_data);
  assert(bias_data);

  Tensor input{input_shape, const_cast<float *>(input_data)};
  Tensor bias{bias_shape, const_cast<float *>(bias_data)};
  Tensor output{output_shape, output_data};

  functor::BiasActivation<Device, float> functor;
  const Eigen::ThreadPoolDevice &d = *eigen_support::GetThreadPoolDevice();
  functor(d, static_cast<const Tensor &>(input).flat<float>(),
          static_cast<const Tensor &>(bias).flat<float>(), output.flat<float>(), activation_min,
          activation_max);
}

} // namespace bias_op
} // namespace cker
} // namespace nnfw
#endif // __NNFW_CKER_EIGEN_BIAS_OP_H__
