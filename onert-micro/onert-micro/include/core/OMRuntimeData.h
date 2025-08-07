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

#include "OMTensorData.h"
#include "OMUtils.h"

#include <memory>
#include <type_traits>

namespace onert_micro
{
namespace core
{

// clang-format off

// ------------------------------------------------------------------------------------------------

class OMBaseContext
{
protected:
  const circle::Tensor *_tensor;
  OMRuntimeShape _shape;

public:
  explicit OMBaseContext(const circle::Tensor *tensor)
    : _tensor(tensor)
    , _shape(tensor)
  {
    assert(_tensor != nullptr);
  }

  virtual ~OMBaseContext() = default;

public:
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

  size_t DimsCount() const
  {
    return _shape.dimensionsCount();
  }

  size_t ShapeFlatSize() const
  {
    return _shape.flatSize();
  }
};

// ------------------------------------------------------------------------------------------------

template <typename T, size_t InputTensorIdx = 0>
class OMInputContext : public OMBaseContext
{
protected:
  OMTensorData<const T> _data;

public:
  template <class RuntimeKernel>
  explicit OMInputContext(const RuntimeKernel &rtk)
    : OMBaseContext(rtk.inputs[InputTensorIdx])
    , _data(MakeInputData<T>(rtk, InputTensorIdx))
  {
    assert(!_data.IsNull());
  }

  ~OMInputContext() override = default;

public:
  const OMTensorData<const T>& Data() const noexcept
  {
    return _data;
  }
};

// ------------------------------------------------------------------------------------------------

template <class T, size_t OutputTensorIdx = 0>
class OMOutputContext : public OMBaseContext
{
protected:
  OMTensorData<T> _data;

public:
  template <class RuntimeKernel>
  explicit OMOutputContext(const RuntimeKernel &rtk)
    : OMBaseContext(rtk.outputs[OutputTensorIdx])
    , _data(MakeOutputData<T>(rtk, OutputTensorIdx))
  {
    assert(!_data.IsNull());
  }

  ~OMOutputContext() override = default;

public:
  OMTensorData<T> &Data() noexcept
  {
    return _data;
  }
};

// ------------------------------------------------------------------------------------------------

template <class T, class ... Mixins>
class OMDataContext : public Mixins...
{
protected:
  OMInputContext<T> _in_ctx;
  OMOutputContext<T> _out_ctx;

public:
  template <class RuntimeKernel>
  explicit OMDataContext(RuntimeKernel &rt_kernel)
    : Mixins(rt_kernel)... 
    , _in_ctx(rt_kernel)
    , _out_ctx(rt_kernel)
  {}

  virtual ~OMDataContext() = default;

public:
  auto &Input()
  {
    return _in_ctx;
  }

  auto &Output()
  {
    return _out_ctx;
  }
};

// ------------------------------------------------------------------------------------------------

template <size_t AxisTensorIdx = 1>
class OMAxisContextMixin
{
public:
  using OMAxisContext = OMInputContext<int32_t, AxisTensorIdx>;

protected:
  OMAxisContext _axis_ctx;

public:
  template <class RuntimeKernel>
  explicit OMAxisContextMixin(RuntimeKernel &rt_kernel)
    : _axis_ctx(rt_kernel)
  {}

public:
  OMAxisContext &Axis()
  {
    return _axis_ctx;
  }
};

// ------------------------------------------------------------------------------------------------

} // namespace core
} // namespace onert_micro

#endif // ONERT_MICRO_CORE_RUNTIME_DATA_H
