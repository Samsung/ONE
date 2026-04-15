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

#ifndef __NNKIT_TENSOR_CONTEXT_H__
#define __NNKIT_TENSOR_CONTEXT_H__

#include <nncc/core/ADT/tensor/Shape.h>
#include <nncc/core/ADT/tensor/Reader.h>
#include <nncc/core/ADT/tensor/Accessor.h>

#include <string>
#include <functional>
#include <stdexcept>
#include <cstdint>

namespace nnkit
{

// NOTE This interface is subject to change.
struct TensorContext
{
  template <typename T>
  using TypedReader = std::function<void(const TensorContext &, uint32_t n,
                                         const nncc::core::ADT::tensor::Reader<T> &)>;

  template <typename T>
  using TypedAccessor =
    std::function<void(const TensorContext &, uint32_t n, nncc::core::ADT::tensor::Accessor<T> &)>;

  virtual ~TensorContext() = default;

  // The number of tensors that this context provides
  virtual uint32_t size(void) const = 0;

  // Query on properties of each tensor
  virtual std::string name(uint32_t n) const = 0;
  virtual nncc::core::ADT::tensor::Shape shape(uint32_t n) const = 0;

  // TODO Support generic byte tensor
  // TODO Support typed tensor for primitive types such as half(fp16), double(fp64), int8(s8),
  // uint8(u8), uint(u32)

  // Float (fp32) tensor support
  virtual bool isFloatTensor(uint32_t n) const
  {
    throw std::runtime_error("This method should be overriden");
  }

  virtual void getMutableFloatTensor(uint32_t n, const TypedAccessor<float> &cb)
  {
    throw std::runtime_error("This method should be overriden");
  }

  virtual void getConstFloatTensor(uint32_t n, const TypedReader<float> &cb) const
  {
    throw std::runtime_error("This method should be overriden");
  }

  // S32
  virtual bool isS32Tensor(uint32_t n) const
  {
    throw std::runtime_error("This method should be overriden");
  }

  virtual void getMutableS32Tensor(uint32_t n, const TypedAccessor<int32_t> &cb)
  {
    throw std::runtime_error("This method should be overriden");
  }

  virtual void getConstS32Tensor(uint32_t n, const TypedReader<int32_t> &cb) const
  {
    throw std::runtime_error("This method should be overriden");
  }
};

} // namespace nnkit

#endif // __NNKIT_TENSOR_CONTEXT_H__
