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

#ifndef _MIR_TENSOR_H_
#define _MIR_TENSOR_H_

#include "mir/ExternalRegion.h"
#include "mir/TensorVariant.h"

namespace mir
{

template <typename T> class Tensor final
{
public:
  explicit Tensor(const TensorVariant &t) : _proxy(t) {}

  T at(const Index &id) const { return *reinterpret_cast<T *>(this->_proxy.at(id)); }

  T &at(const Index &id) { return *reinterpret_cast<T *>(this->_proxy.at(id)); }

  T atOffset(int32_t offset) const { return *reinterpret_cast<T *>(this->_proxy.atOffset(offset)); }

  T &atOffset(int32_t offset) { return *reinterpret_cast<T *>(this->_proxy.atOffset(offset)); }

  ExternalRegion<T> getRegion(const Index &idx)
  {
    // Only last dimension is safe to process continiously
    auto last_dim = getShape().rank() - 1;
    auto base = reinterpret_cast<T *>(_proxy.at(idx));
    auto length = getShape().dim(last_dim) - idx.at(last_dim);
    return ExternalRegion<T>(base, length);
  }

  const Shape &getShape() const { return _proxy.getShape(); };

private:
  const TensorVariant &_proxy;
};

extern template class Tensor<float>;

extern template class Tensor<double>;

extern template class Tensor<int>;

} // namespace mir

#endif //_MIR_TENSOR_H_
