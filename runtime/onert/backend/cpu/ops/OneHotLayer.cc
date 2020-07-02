/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "OneHotLayer.h"

#include "OperationUtils.h"

#include <cker/operation/OneHot.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{

template <typename T> void OneHotLayer::oneHotImpl()
{
  // It assumes index is int32_t type.
  nnfw::cker::OneHot<T, int32_t>(
      *reinterpret_cast<const int32_t *>(_depth->buffer()),
      *reinterpret_cast<T *>(_on_value->buffer()), *reinterpret_cast<T *>(_off_value->buffer()),
      _axis, getTensorShape(_indices), reinterpret_cast<const int32_t *>(_indices->buffer()),
      getTensorShape(_output), reinterpret_cast<T *>(_output->buffer()));
}

void OneHotLayer::configure(const IPortableTensor *indices, const IPortableTensor *depth,
                            const IPortableTensor *on_value, const IPortableTensor *off_value,
                            IPortableTensor *output, const int32_t axis)
{
  _indices = indices;
  _output = output;
  _depth = depth;
  _on_value = on_value;
  _off_value = off_value;
  _axis = axis;
}

void OneHotLayer::run()
{
  if (_output->data_type() == OperandType::FLOAT32)
  {
    oneHotImpl<float>();
  }
  else
  {
    throw std::runtime_error{"OneHot: unsupported data type"};
  }
}

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
