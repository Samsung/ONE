/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __NNKIT_SUPPORT_CAFFE_TENSOR_CONTEXT_H__
#define __NNKIT_SUPPORT_CAFFE_TENSOR_CONTEXT_H__

#include "nnkit/support/caffe/BlobContext.h"

#include <nnkit/TensorContext.h>

#include <nncc/core/ADT/tensor/LexicalLayout.h>
#include <nncc/core/ADT/tensor/Overlay.h>

#include <type_traits>
#include <stdexcept>

namespace nnkit
{
namespace support
{
namespace caffe
{

template <typename DType> class TensorContext final : public nnkit::TensorContext
{
public:
  TensorContext(BlobContext<DType> &blobs) : _blobs(blobs)
  {
    // DO NOTHING
  }

private:
  static nncc::core::ADT::tensor::Shape shapeOf(const ::caffe::Blob<DType> &blob)
  {
    nncc::core::ADT::tensor::Shape shape;

    const uint32_t rank = blob.shape().size();

    shape.resize(rank);
    for (uint32_t axis = 0; axis < rank; ++axis)
    {
      shape.dim(axis) = blob.shape(axis);
    }

    return shape;
  }

public:
  uint32_t size(void) const override { return _blobs.size(); }

  std::string name(uint32_t n) const override { return _blobs.name(n); }

  nncc::core::ADT::tensor::Shape shape(uint32_t n) const override
  {
    return shapeOf(*_blobs.blob(n));
  }

  // Float (fp32) tensor support
  bool isFloatTensor(uint32_t n) const override { return std::is_same<DType, float>::value; }

  void getMutableFloatTensor(uint32_t n, const TensorContext::TypedAccessor<float> &f) override
  {
    if (!std::is_same<DType, float>::value)
    {
      throw std::runtime_error{"type mismatch"};
    }

    using nncc::core::ADT::tensor::LexicalLayout;
    using nncc::core::ADT::tensor::make_overlay;

    auto base = _blobs.region(n);
    auto view = make_overlay<float, LexicalLayout>(shape(n), base);

    f(*this, n, view);
  }

  void getConstFloatTensor(uint32_t n, const TensorContext::TypedReader<float> &f) const override
  {
    if (!std::is_same<DType, float>::value)
    {
      throw std::runtime_error{"type mismatch"};
    }

    using nncc::core::ADT::tensor::LexicalLayout;
    using nncc::core::ADT::tensor::make_overlay;

    auto base = _blobs.region(n);
    auto view = make_overlay<float, LexicalLayout>(shape(n), base);

    f(*this, n, view);
  }

private:
  BlobContext<DType> &_blobs;
};

} // namespace caffe
} // namespace support
} // namespace nnkit

#endif // __NNKIT_SUPPORT_CAFFE_TENSOR_CONTEXT_H__
