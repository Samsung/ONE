/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __ONERT_BACKEND_GPU_CL_INTERNAL_TENSOR_H__
#define __ONERT_BACKEND_GPU_CL_INTERNAL_TENSOR_H__

#include <stdint.h>

#include <vector>

#include "DataType.h"
#include "Shape.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{
namespace internal_tensor
{

// Meta function given element type returns a type for Tensor data container.
template <DataType Type> struct StorageType;

template <> struct StorageType<DataType::FLOAT32>
{
  using value = std::vector<float>;
};

template <> struct StorageType<DataType::INT32>
{
  using value = std::vector<int32_t>;
};

} // namespace internal_tensor

template <typename ShapeT, DataType Type> struct InternalTensor
{
  using ShapeType = ShapeT;

  constexpr static DataType kType = Type;

  using TensorStorageType = typename internal_tensor::StorageType<Type>::value;

  // Opaque id of a tensor.
  int64_t id = -1;

  ShapeType shape;

  TensorStorageType data;
};

// TensorRef is a reference to another tensor. If an object should never hold
// tensor data, then TensorRef should be used instead.
template <typename ShapeT> struct TensorRef
{
  using ShapeType = ShapeT;

  DataType type = DataType::UNKNOWN;

  ShapeT shape;

  // Opaque reference to a tensor. Upstream component is responsible for
  // resolving this reference into an actual tensor.
  int64_t ref = -1;

  // Specifies if the tensor should be a variable input tensor that must be an
  // output as well as an input to the graph.
  bool is_variable_input = false;
};

template <typename ShapeT, DataType Type> constexpr DataType InternalTensor<ShapeT, Type>::kType;

template <typename ShapeT, DataType Type>
InternalTensor<ShapeT, Type> MakeZeroTensor(const ShapeT &shape)
{
  InternalTensor<ShapeT, Type> tensor;
  tensor.shape = shape;
  tensor.data =
    typename InternalTensor<ShapeT, Type>::TensorStorageType(shape.DimensionsProduct(), 0);
  return tensor;
}

using TensorFloat32 = InternalTensor<BHWC, DataType::FLOAT32>;
using Tensor5DFloat32 = InternalTensor<BHWDC, DataType::FLOAT32>;

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_GPU_CL_INTERNAL_TENSOR_H__
