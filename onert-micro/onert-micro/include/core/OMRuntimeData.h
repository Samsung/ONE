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
#ifndef ONERT_MICRO_CORE_RUNTIME_DATA_H
#define ONERT_MICRO_CORE_RUNTIME_DATA_H

#include "OMCustomTensorData.h"
#include "OMUtils.h"

namespace onert_micro::core
{

// clang-format off

// ------------------------------------------------------------------------------------------------

template <typename T, class TensorData>
class OMTensorContext
{
protected:
  TensorData _data;
  const circle::Tensor *_tensor;
  OMRuntimeShape _shape;

public:
  OMTensorContext(TensorData &&data, const circle::Tensor *tensor)
    : _data(std::move(data))
    , _tensor(tensor)
    , _shape(tensor)
  {
    assert(_tensor != nullptr);
  }

public:
  TensorData &Data()
  {
    return _data;
  }

  const TensorData &Data() const
  {
    return _data;
  }

  const circle::Tensor *Tensor() const noexcept
  {
    return _tensor;
  }

  const OMRuntimeShape &Shape() const noexcept
  {
    return _shape;
  }

  const int32_t *Dims() const
  {
    return _shape.dimsData();
  }

  size_t DimLength(size_t idx) const
  {
    auto result = _shape.dims(idx);
    assert(result >= 0);

    return static_cast<size_t>(result);
  }

  bool HasZeroSizeDims() const
  {
    return _shape.hasZeroSizeDims();
  }

  size_t DimsCount() const
  {
    return _shape.dimensionsCount();
  }

  size_t ElementsCount() const
  {
    return _shape.flatSize();
  }

  bool IsScalar() const noexcept
  {
    return _shape.isScalar();
  }
};

// ------------------------------------------------------------------------------------------------

template <typename T>
using OMInputContext = OMTensorContext<const T, IOTensorData<const T>>;

template <typename T>
using OMOutputContext = OMTensorContext<T, IOTensorData<T>>;

using OMAxisContext = OMTensorContext<uint32_t, OMAxisData<uint32_t>>;

// ------------------------------------------------------------------------------------------------

template <typename T, class RuntimeKernel>
OMInputContext<T> MakeInputContext(RuntimeKernel &rtk, size_t inputIdx = 0)
{
  auto data = MakeInputData<T>(rtk, inputIdx);
  const circle::Tensor *tensor = rtk.inputs[inputIdx];
  return OMInputContext<T>(std::move(data), tensor);
}

template <typename T, class RuntimeKernel>
OMOutputContext<T> MakeOutputContext(RuntimeKernel &rtk, size_t outputIdx = 0)
{
  auto data = MakeOutputData<T>(rtk, outputIdx);
  const circle::Tensor *tensor = rtk.outputs[outputIdx];
  return OMOutputContext<T>(std::move(data), tensor);
}

template <class RuntimeKernel>
OMAxisContext MakeAxisContext(RuntimeKernel &rtk, size_t axisInputIdx = 1)
{
  auto data = MakeAxisData<uint32_t>(rtk, axisInputIdx);
  const circle::Tensor *tensor = rtk.inputs[axisInputIdx];
  return OMAxisContext(std::move(data), tensor);
}

// ------------------------------------------------------------------------------------------------

template <class T, class ... Mixins>
class OMDataContext : public Mixins...
{
protected:
  OMInputContext<T> _input;
  OMOutputContext<T> _output;

public:
  template <class RuntimeKernel>
  explicit OMDataContext(RuntimeKernel &rt_kernel)
    : Mixins(rt_kernel)...
    , _input(MakeInputContext<T>(rt_kernel))
    , _output(MakeOutputContext<T>(rt_kernel))
  {}

  virtual ~OMDataContext() = default;

public:
  auto &Input()
  {
    return _input;
  }

  auto &Output()
  {
    return _output;
  }
};

// ------------------------------------------------------------------------------------------------

template <typename IntType = uint32_t, size_t AxisTensorIdx = 1>
class OMAxisContextMixin
{
protected:
  OMAxisContext _axis;

public:
  template <class RuntimeKernel>
  explicit OMAxisContextMixin(RuntimeKernel &rt_kernel)
    : _axis(MakeAxisContext(rt_kernel, AxisTensorIdx))
  {}

public:
  OMAxisContext &Axis()
  {
    return _axis;
  }
};

// ------------------------------------------------------------------------------------------------

} // namespace onert_micro::core

#endif // ONERT_MICRO_CORE_RUNTIME_DATA_H
