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

// A list of tensors in a format that can be used by kernels like split and
// concatenation.
template <typename T> class VectorOfTensors
{
public:
  // Build with the tensors in 'tensor_list'.
  explicit VectorOfTensors(const std::vector<const Tensor *> &tensor_list)
  {
    const int num_tensors = tensor_list.size();

    all_data_.reserve(num_tensors);
    all_shape_.reserve(num_tensors);
    all_shape_ptr_.reserve(num_tensors);

    for (const Tensor *tensor : tensor_list)
    {
      all_data_.push_back(getTensorData<T>(tensor));
      all_shape_.push_back(getTensorShape(tensor));
    }

    // Taking the pointer from inside a std::vector is only OK if the vector is
    // never modified, so we populate all_shape in the previous loop and then we
    // are free to grab iterators here.
    for (tflite::RuntimeShape &shape : all_shape_)
    {
      all_shape_ptr_.push_back(&shape);
    }
  }
  // Return a pointer to the data pointers of all tensors in the list. For
  // example:
  //   float* const* f = v.data();
  //   f[0][1] is the second element of the first tensor.
  const T *const *data() const { return all_data_.data(); }

  // Return a pointer the shape pointers of all tensors in the list. For
  // example:
  //   const RuntimeShape* const* d = v.dims();
  //   dims[1] are the dimensions of the second tensor in the list.
  const tflite::RuntimeShape *const *shapes() const { return all_shape_ptr_.data(); }

private:
  std::vector<const T *> all_data_;
  std::vector<tflite::RuntimeShape> all_shape_;
  std::vector<tflite::RuntimeShape *> all_shape_ptr_;
};

// A list of quantized tensors in a format that can be used by kernels like
// split and concatenation.
class VectorOfQuantizedTensors : public VectorOfTensors<uint8_t>
{
public:
  // Build with the tensors in 'tensor_list'.
  explicit VectorOfQuantizedTensors(const std::vector<const Tensor *> &tensor_list)
      : VectorOfTensors<uint8_t>(tensor_list)
  {
    for (const Tensor *tensor : tensor_list)
    {
      zero_point_.push_back(tensor->zero_point());
      scale_.push_back(tensor->scale());
    }
  }

  const float *scale() const { return scale_.data(); }
  const int32_t *zero_point() const { return zero_point_.data(); }

private:
  std::vector<int32_t> zero_point_;
  std::vector<float> scale_;
};

} // namespace kernels
} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_KERNELS_UTILS_H
