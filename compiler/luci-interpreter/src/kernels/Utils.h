/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef LUCI_INTERPRETER_KERNELS_UTILS_H
#define LUCI_INTERPRETER_KERNELS_UTILS_H

#include "core/KernelParams.h"
#include "core/Tensor.h"

#include <tensorflow/lite/kernels/internal/types.h>

namespace luci_interpreter
{
namespace kernels
{

void calculateActivationRange(Activation activation, float *activation_min, float *activation_max);

inline tflite::RuntimeShape getTensorShape(const Tensor *tensor)
{
  if (tensor == nullptr)
    return tflite::RuntimeShape();

  const Shape &shape = tensor->shape();
  tflite::RuntimeShape runtime_shape(shape.num_dims());
  for (int i = 0; i < shape.num_dims(); ++i)
  {
    runtime_shape.SetDim(i, shape.dim(i));
  }
  return runtime_shape;
}

template <typename T> const T *getTensorData(const Tensor *tensor)
{
  return tensor != nullptr ? tensor->data<T>() : nullptr;
}

template <typename T> T *getTensorData(Tensor *tensor)
{
  return tensor != nullptr ? tensor->data<T>() : nullptr;
}

} // namespace kernels
} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_KERNELS_UTILS_H
