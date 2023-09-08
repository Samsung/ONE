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

#ifndef __NNFW_CKER_TRAIN_OPTIMIZER_ADAM_H__

#include "cker/eigen/training_ops.h"
#include "cker/eigen/EigenSupport.h"

#include <vector>

namespace nnfw
{
namespace cker
{
namespace train
{

inline void Adam(const Shape &trainable_shape, float *trainable_data, const Shape &grad_shape,
                 const float *grad_data, const Shape &m_shape, float *m_data, const Shape &v_shape,
                 float *v_data, float beta1_power, float beta2_power, float learning_rate,
                 float beta1, float beta2, float epsilon, bool use_nesterov)
{
  Tensor trainable_tensor;
  Tensor grad_tensor;
  Tensor m_tensor;
  Tensor v_tensor;
  Tensor beta1_power_tensor;
  Tensor beta2_power_tensor;
  Tensor lr_tensor;
  Tensor beta1_tensor;
  Tensor beta2_tensor;
  Tensor epsilon_tensor;

  trainable_tensor.shape.ReplaceWith(trainable_shape.DimensionsCount(), trainable_shape.DimsData());
  trainable_tensor.buffer = trainable_data;

  grad_tensor.shape.ReplaceWith(grad_shape.DimensionsCount(), grad_shape.DimsData());
  grad_tensor.buffer = const_cast<float *>(grad_data);

  m_tensor.shape.ReplaceWith(m_shape.DimensionsCount(), m_shape.DimsData());
  m_tensor.buffer = m_data;

  v_tensor.shape.ReplaceWith(v_shape.DimensionsCount(), v_shape.DimsData());
  v_tensor.buffer = v_data;

  std::vector<float> beta1_power_vec{beta1_power};
  beta1_power_tensor.buffer = beta1_power_vec.data();

  std::vector<float> beta2_power_vec{beta2_power};
  beta2_power_tensor.buffer = beta2_power_vec.data();

  std::vector<float> lr_vec{learning_rate};
  lr_tensor.buffer = lr_vec.data();

  std::vector<float> beta1_vec{beta1};
  beta1_tensor.buffer = beta1_vec.data();

  std::vector<float> beta2_vec{beta2};
  beta2_tensor.buffer = beta2_vec.data();

  std::vector<float> epsilon_vec{epsilon};
  epsilon_tensor.buffer = epsilon_vec.data();

  if (trainable_shape != m_shape)
    throw std::runtime_error("cker::Adam: output and m do not have the same shape");

  if (trainable_shape != v_shape)
    throw std::runtime_error("cker::Adam: output and v do not have the same shape");

  if (trainable_shape != grad_shape)
    throw std::runtime_error("cker::Adam: output and gradient do not have the same shape");

  const training_ops::CPUDevice &device = *eigen_support::GetThreadPoolDevice();
  training_ops::functor::ApplyAdam<training_ops::CPUDevice, float>()(
    device, trainable_tensor.flat<float>(), m_tensor.flat<float>(), v_tensor.flat<float>(),
    beta1_power_tensor.scalar<float>(), beta2_power_tensor.scalar<float>(),
    lr_tensor.scalar<float>(), beta1_tensor.scalar<float>(), beta2_tensor.scalar<float>(),
    epsilon_tensor.scalar<float>(), static_cast<const Tensor &>(grad_tensor).flat<float>(),
    use_nesterov);
}

} // namespace train
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_TRAIN_OPTIMIZER_ADAM_H__
