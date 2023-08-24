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

#ifndef __NNFW_CKER_TRAIN_OPTIMIZER_SGD_H__

// #include "OptimizerHelpers.h"
#include "cker/eigen/training_ops.h"
#include "cker/eigen/EigenSupport.h"

#include <vector>

namespace nnfw
{
namespace cker
{
namespace train
{

inline void GradientDescent(const Shape &output_shape, float *output_data, const Shape &grad_shape,
                            const float *grad_data, float learning_rate)
{
  Tensor output_tensor;
  Tensor grad_tensor;
  Tensor lr_tensor;

  output_tensor.shape.ReplaceWith(output_shape.DimensionsCount(), output_shape.DimsData());
  output_tensor.buffer = output_data;

  grad_tensor.shape.ReplaceWith(grad_shape.DimensionsCount(), grad_shape.DimsData());
  grad_tensor.buffer = const_cast<float *>(grad_data);

  std::vector<float> lr_vec{learning_rate};
  lr_tensor.buffer = lr_vec.data();

  if (output_shape != grad_shape)
    throw std::runtime_error(
      "cker::GradientDescent: output and gradient do not have the same shape");

  const training_ops::CPUDevice &device = *eigen_support::GetThreadPoolDevice();
  training_ops::functor::ApplyGradientDescent<training_ops::CPUDevice, float>()(
    device, output_tensor.flat<float>(), lr_tensor.scalar<float>(),
    static_cast<const Tensor &>(grad_tensor).flat<float>());
}

} // namespace train
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_TRAIN_OPTIMIZER_SGD_H__
