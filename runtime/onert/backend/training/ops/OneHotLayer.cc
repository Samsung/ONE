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
namespace training
{
namespace ops
{

template <typename T> void OneHotLayer::oneHotImpl()
{
  // It assumes index is int32_t type.
  nnfw::cker::OneHot<T, int32_t>(
    *getBuffer<int32_t>(_depth), *getBuffer<T>(_on_value), *getBuffer<T>(_off_value), _axis,
    getShape(_indices), getBuffer<int32_t>(_indices), getShape(_output), getBuffer<T>(_output));
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
} // namespace training
} // namespace backend
} // namespace onert
