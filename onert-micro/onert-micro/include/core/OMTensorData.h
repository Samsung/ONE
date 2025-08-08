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

#include "OMQuantizationData.h"
#include "OMUtils.h"

#include <map>
#include <memory>
#include <type_traits>

namespace onert_micro
{
namespace core
{

// clang-format off

// ------------------------------------------------------------------------------------------------

template <typename T>
class OMTensorData
{
  using QuantData = OMQuantizationData<T>;
  using QuantDataPtr = std::unique_ptr<QuantData>;

private:
  T *_data = nullptr;
  size_t _size = 0;
  QuantDataPtr _quant = {};
  std::map<size_t, float> _out_float_data = {};

public:
  OMTensorData(T *data, const circle::Tensor *tensor)
    : _data(data)
  {
    assert(data != nullptr);
    assert(tensor != nullptr);

    _size = OMRuntimeShape(tensor).flatSize();

    if (IsQuantized<T>)
    {
      _quant = std::make_unique<QuantData>(_data, tensor);
    }
  }

public:
  template <typename U = T>
  constexpr static bool IsInt8 = std::is_same<std::decay_t<U>, int8_t>::value;

  template <typename U = T>
  constexpr static bool IsUInt8 = std::is_same<std::decay_t<U>, uint8_t>::value;

  template <typename U = T>
  constexpr static bool IsQuantized = IsInt8<U> || IsUInt8<U>;

public:
  template <typename U = T>
  constexpr static bool HasConstData = std::is_const<U>::value;

  template <typename U = T>
  constexpr static bool HasNonConstData = !HasConstData<U>;

  template <typename U = T>
  constexpr static bool HasQuantizedConstData = IsQuantized<U> && HasConstData<U>;

  template <typename U = T>
  constexpr static bool HasQuantizedNonConstData = IsQuantized<U> && HasNonConstData<U>;

public:
  bool IsNull() const noexcept
  {
    return (_data == nullptr) || (_size == 0);
  }

public:
  T* Get() const noexcept
  {
    return _data;
  }

  T& At(size_t idx) const
  {
    CheckIndex(idx);
    return _data[idx];
  }

  void SetAt(size_t idx, T value)
  {
    CheckIndex(idx);
    _data[idx] = value;
  }

  template <typename U = T>
  std::enable_if_t<IsQuantized<U>> SetAt(size_t idx, float value)
  {
    CheckIndex(idx);
    _quant->SetDataAt(idx, value);
  }

public:
  template <typename U = T>
  std::enable_if_t<!IsQuantized<U>, T&> ValueAt(size_t idx) const
  {
    return At(idx);
  }

  template <typename U = T>
  std::enable_if_t<HasQuantizedConstData<U>, float> ValueAt(size_t idx) const
  {
    CheckIndex(idx);
    return _quant->DataAt(idx);
  }

  template <typename U = T>
  std::enable_if_t<HasQuantizedNonConstData<U>, float> ValueAt(size_t idx) const
  {
    CheckIndex(idx);

    if (_out_float_data.count(idx) == 0)
      return 0.f;

    return _out_float_data.at(idx);
  }

public:
  template <typename U = T>
  std::enable_if_t<!IsQuantized<U>> SetValueAt(size_t idx, T value)
  {
    SetAt(idx, value);
  }

  template <typename U = T>
  std::enable_if_t<HasQuantizedNonConstData<U>> SetValueAt(size_t idx, float value)
  {
    CheckIndex(idx);
    _out_float_data[idx] = value;
  }

private:
  bool CheckIndex(size_t idx) const
  {
    assert(idx < _size);
    return idx < _size;
  }
};

// ------------------------------------------------------------------------------------------------

template <class T, class RuntimeKernel>
OMTensorData<const T> MakeInputData(RuntimeKernel& rtk, size_t input_idx)
{
  const T *data = utils::castInputData<T>(rtk.inputs_data[input_idx]);
  const circle::Tensor *tensor = rtk.inputs[input_idx];

  assert(data != nullptr);
  assert(tensor != nullptr);

  OMTensorData<const T> result(data, tensor);
  return result;
}

template <class T, class RuntimeKernel>
OMTensorData<T> MakeOutputData(RuntimeKernel& rtk, size_t output_idx)
{
  T *data = utils::castOutputData<T>(rtk.outputs_data[output_idx]);
  const circle::Tensor *tensor = rtk.outputs[output_idx];

  assert(data != nullptr);
  assert(tensor != nullptr);

  OMTensorData<T> result(data, tensor);
  return result;
}

// ------------------------------------------------------------------------------------------------

} // namespace core
} // namespace onert_micro

#endif // ONERT_MICRO_CORE_TENSOR_DATA_H
