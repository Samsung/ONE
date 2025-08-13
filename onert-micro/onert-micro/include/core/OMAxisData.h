/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
#ifndef ONERT_MICRO_CORE_AXIS_DATA_H
#define ONERT_MICRO_CORE_AXIS_DATA_H

#include "OMQuantizationData.h"
#include "OMRuntimeShape.h"
#include "OMUtils.h"

namespace onert_micro::core
{

// clang-format off

// ------------------------------------------------------------------------------------------------

class OMAxisData
{
  int32_t *_data = nullptr;
  size_t _size = 0;

public:
  OMAxisData(int32_t *data, const circle::Tensor *tensor)
    : _data(data)
  {
    assert(data != nullptr);
    assert(tensor != nullptr);

    _size = OMRuntimeShape(tensor).flatSize();
  }

public:
  bool IsNull() const noexcept
  {
    return (_data == nullptr) || (_size == 0);
  }

public:
  int32_t* Get() const noexcept
  {
    return _data;
  }

  int32_t& At(size_t idx) const
  {
    CheckIndex(idx);
    return _data[idx];
  }

  void SetAt(size_t idx, int32_t value)
  {
    CheckIndex(idx);
    _data[idx] = value;
  }

public:
  int32_t& AxisAt(size_t idx)
  {
    CheckIndex(idx);
    return _data[idx];
  }

  int32_t PositiveAxisAt(size_t idx)
  {
    return AxisIndexOffset(idx, _data[idx] < 0, 1);
  }

  int32_t NegativeAxisAt(size_t idx)
  {
    return AxisIndexOffset(idx, _data[idx] > 0, -1);
  }

private:
  // Handle negative index. A positive index 'p_idx' can be represented as a
  // negative index 'n_idx' as: n_idx = p_idx - num_dims
  // eg: For num_dims=3, [0, 1, 2] is the same as [-3, -2, -1]

  int AxisIndexOffset(size_t idx, bool needsOffset, int factor)
  {
    if (!needsOffset)
      return At(idx);

    return At(idx) + factor * _size;
  }

  bool CheckIndex(size_t idx) const
  {
    if (idx >= 0)
    {
      assert(idx < _size);
      return idx < _size;
    }

    assert(idx >= _size);
    return idx >= _size;
  }
};

// ------------------------------------------------------------------------------------------------

template <class RuntimeKernel>
OMAxisData MakeAxisData(RuntimeKernel& rtk, size_t input_idx)
{
  int32_t *data = reinterpret_cast<int32_t*>(rtk.inputs_data[input_idx]);
  const circle::Tensor *tensor = rtk.inputs[input_idx];

  assert(data != nullptr);
  assert(tensor != nullptr);

  OMAxisData result(data, tensor);
  return result;
}

// ------------------------------------------------------------------------------------------------

} // namespace onert_micro::core

#endif // ONERT_MICRO_CORE_AXIS_DATA_H
