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
#ifndef ONERT_MICRO_CORE_CUSTOM_TENSOR_DATA_H
#define ONERT_MICRO_CORE_CUSTOM_TENSOR_DATA_H

#include "OMQuantizationData.h"
#include "OMTensorData.h"
#include "OMTypeTraits.h"

#include <memory>

// clang-format off

namespace onert_micro::core
{

using type_traits::EnableIfQuantized;
using type_traits::EnableIfNotQuantized;
using type_traits::IsQuantized;

// ------------------------------------------------------------------------------------------------

template <typename T>
class IOTensorData : public OMTensorData<T>
{
  using QuantData = OMQuantizationData<T>;
  using QuantDataPtr = std::unique_ptr<QuantData>;

private:
  QuantDataPtr _quant = {};

public:
  IOTensorData(T *data, const circle::Tensor *tensor)
    : OMTensorData<T>(data, tensor)
  {
    if (IsQuantized<T>)
    {
      _quant = std::make_unique<QuantData>(this->_data, tensor);
    }
  }

public:
  template <typename U = T>
  EnableIfNotQuantized<U, U> ValueAt(size_t idx) const
  {
    return this->At(idx);
  }

  template <typename U = T>
  EnableIfQuantized<U, float> ValueAt(size_t idx) const
  {
    this->CheckIndex(idx);
    return _quant->DataAt(idx);
  }

public:
  template <typename U = T>
  EnableIfNotQuantized<U, void> SetValueAt(size_t idx, T value)
  {
    this->SetAt(idx, value);
  }

  template <typename U = T>
  EnableIfQuantized<U, void> SetValueAt(size_t idx, float value)
  {
    this->CheckIndex(idx);
    _quant->SetDataAt(idx, value);
  }
};

// ------------------------------------------------------------------------------------------------

template <typename IntType = uint32_t>
class OMAxisData : public OMTensorData<int32_t, IntType>
{
  std::vector<IntType> _axes = {};

public:
  OMAxisData(int32_t *data, const circle::Tensor *tensor)
    : OMTensorData<int32_t, IntType>(data, tensor)
  {
    /*
      Handle negative index.
      A positive index 'p_idx' can be represented as a negative index 'n_idx' as:
      n_idx = p_idx - num_dims.
      eg: For num_dims=3, [0, 1, 2] is the same as [-3, -2, -1].
    */
    std::transform(data, data + this->_size, std::back_inserter(_axes), [this](auto axis)
    {
      auto offset = this->_size * static_cast<size_t>(axis < 0);
      return static_cast<IntType>(axis + offset);
    });
  }

public:
  IntType *Get() noexcept override
  {
    return _axes.data();
  }

  const IntType *Get() const noexcept override
  {
    return _axes.data();
  }
};

// ------------------------------------------------------------------------------------------------

template <typename T, class RuntimeKernel>
IOTensorData<const T> MakeInputData(RuntimeKernel &rtk, size_t input_idx)
{
  auto data = reinterpret_cast<const T*>(rtk.inputs_data[input_idx]);
  const circle::Tensor *tensor = rtk.inputs[input_idx];

  assert(data != nullptr);
  assert(tensor != nullptr);

  IOTensorData<const T> result(data, tensor);
  return result;
}

template <typename T, class RuntimeKernel>
IOTensorData<T> MakeOutputData(RuntimeKernel &rtk, size_t output_idx)
{
  auto data = reinterpret_cast<T*>(rtk.outputs_data[output_idx]);
  const circle::Tensor *tensor = rtk.outputs[output_idx];

  assert(data != nullptr);
  assert(tensor != nullptr);

  IOTensorData<T> result(data, tensor);
  return result;
}

template <typename IntType, class RuntimeKernel>
OMAxisData<IntType> MakeAxisData(RuntimeKernel &rtk, size_t axisTensorIdx)
{
  auto data = reinterpret_cast<int32_t*>(rtk.inputs_data[axisTensorIdx]);
  const circle::Tensor *tensor = rtk.inputs[axisTensorIdx];

  assert(data != nullptr);
  assert(tensor != nullptr);

  OMAxisData<IntType> result(data, tensor);
  return result;
}

// ------------------------------------------------------------------------------------------------

} // namespace onert_micro::core

#endif // ONERT_MICRO_CORE_CUSTOM_TENSOR_DATA_H
