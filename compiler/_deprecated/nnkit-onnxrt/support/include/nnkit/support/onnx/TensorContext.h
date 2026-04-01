/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __NNKIT_SUPPORT_ONNX_TENSOR_CONTEXT_H__
#define __NNKIT_SUPPORT_ONNX_TENSOR_CONTEXT_H__

#include "nnkit/support/onnx/TensorSet.h"

#include <nnkit/TensorContext.h>

#include <nncc/core/ADT/tensor/LexicalLayout.h>
#include <nncc/core/ADT/tensor/Overlay.h>

namespace nnkit
{
namespace support
{
namespace onnx
{

class TensorContext final : public nnkit::TensorContext
{
public:
  TensorContext(TensorSet &tensors) : _tensors(tensors)
  {
    // DO NOTHING
  }

  uint32_t size(void) const override { return _tensors.size(); }

  std::string name(uint32_t n) const override { return std::string{_tensors.name(n)}; }

  nncc::core::ADT::tensor::Shape shape(uint32_t n) const override
  {
    const std::vector<size_t> &dims = _tensors.dim(n);

    nncc::core::ADT::tensor::Shape shape;
    shape.resize(dims.size());
    for (size_t i = 0; i < dims.size(); ++i)
    {
      shape.dim(i) = dims[i];
    }
    return shape;
  }

  bool isFloatTensor(uint32_t n) const override
  {
    return (_tensors.type(n) == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  }

  void getMutableFloatTensor(uint32_t n, const TensorContext::TypedAccessor<float> &f) override
  {
    if (_tensors.type(n) != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    {
      throw std::runtime_error{"type mismatch"};
    }

    using nncc::core::ADT::tensor::LexicalLayout;
    using nncc::core::ADT::tensor::make_overlay;

    Status status;

    OrtValue *base = _tensors.mutable_tensor(n);
    float *data;

    status = OrtGetTensorMutableData(base, (void **)&data);
    status.throwOnError();

    auto overlay = make_overlay<float, LexicalLayout>(shape(n), data);

    f(*this, n, overlay);
  }

  void getConstFloatTensor(uint32_t n, const TensorContext::TypedReader<float> &f) const override
  {
    if (_tensors.type(n) != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    {
      throw std::runtime_error{"type mismatch"};
    }

    using nncc::core::ADT::tensor::LexicalLayout;
    using nncc::core::ADT::tensor::make_overlay;

    Status status;

    OrtValue *base = _tensors.mutable_tensor(n);
    float *data;

    status = OrtGetTensorMutableData(base, (void **)&data);
    status.throwOnError();

    auto overlay = make_overlay<float, LexicalLayout>(shape(n), data);

    f(*this, n, overlay);
  }

private:
  TensorSet &_tensors;
};

} // namespace onnx
} // namespace support
} // namespace nnkit

#endif // __NNKIT_SUPPORT_ONNX_TENSOR_CONTEXT_H__
