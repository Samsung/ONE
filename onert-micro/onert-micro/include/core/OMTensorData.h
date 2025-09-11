/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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
#ifndef ONERT_MICRO_CORE_TENSOR_DATA_H
#define ONERT_MICRO_CORE_TENSOR_DATA_H

#include "OMRuntimeShape.h"
#include "OMTypeTraits.h"

#include <vector>

namespace onert_micro::core
{

// clang-format off

using type_traits::EnableIfNotConst;

// ------------------------------------------------------------------------------------------------

template <typename TData, typename TValue = TData>
class OMTensorData
{
protected:
  TData *_data = nullptr;
  size_t _size = 0;

public:
  OMTensorData(TData *data, const circle::Tensor *tensor)
  {
    assert(data != nullptr);
    assert(tensor != nullptr);

    _data = data;
    _size = OMRuntimeShape(tensor).flatSize();
  }

public:
  size_t Size() const noexcept
  {
    return _size;
  }

  bool IsNull() const noexcept
  {
    return (_data == nullptr) || (_size == 0);
  }

public:
  template <typename R = TValue>
  EnableIfNotConst<R, R> At(size_t idx)
  {
    CheckIndex(idx);
    auto data = Get();
    return data[idx];
  }

  TValue At(size_t idx) const
  {
    CheckIndex(idx);
    auto data = Get();
    return data[idx];
  }

  template <typename R = TValue>
  EnableIfNotConst<R, void> SetAt(size_t idx, R value)
  {
    CheckIndex(idx);
    auto data = Get();
    data[idx] = value;
  }

public:
  virtual TValue *Get() noexcept
  {
    return reinterpret_cast<TValue*>(_data);
  }

  virtual const TValue *Get() const noexcept
  {
    return reinterpret_cast<const TValue*>(_data);
  }

protected:
  virtual bool CheckIndex(size_t idx) const
  {
    assert(idx < _size);
    return idx < _size;
  }
};

// ------------------------------------------------------------------------------------------------

} // namespace onert_micro::core

#endif // ONERT_MICRO_CORE_TENSOR_DATA_H
