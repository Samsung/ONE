/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/IR/Nodes/CircleRemoteConst.h"

#include <cassert>

namespace luci
{

template <loco::DataType DT> uint32_t CircleRemoteConst::size(void) const
{
  assert(dtype() == DT);
  assert(_buffer.size % sizeof(typename loco::DataTypeImpl<DT>::Type) == 0);
  return _buffer.size / sizeof(typename loco::DataTypeImpl<DT>::Type);
}

template <loco::DataType DT>
const typename loco::DataTypeImpl<DT>::Type &CircleRemoteConst::at(uint32_t n) const
{
  assert(dtype() == DT);
  assert(n < size<DT>());
  return *(reinterpret_cast<const typename loco::DataTypeImpl<DT>::Type *>(_buffer.data) + n);
}

template <loco::DataType DT>
const typename loco::DataTypeImpl<DT>::Type &CircleRemoteConst::scalar(void) const
{
  assert(dtype() == DT);
  return *(reinterpret_cast<const typename loco::DataTypeImpl<DT>::Type *>(_buffer.data));
}

void CircleRemoteConst::bind_buffer(const uint8_t *data, uint32_t size)
{
  assert(size % loco::size(dtype()) == 0);
  _buffer = {data, size};
}

const uint8_t *CircleRemoteConst::data() const { return _buffer.data; }

uint32_t CircleRemoteConst::buffer_size() const { return _buffer.size; }

#define INSTANTIATE(DT)                                                                            \
  template uint32_t CircleRemoteConst::size<DT>(void) const;                                       \
  template const typename loco::DataTypeImpl<DT>::Type &CircleRemoteConst::at<DT>(uint32_t) const; \
  template const typename loco::DataTypeImpl<DT>::Type &CircleRemoteConst::scalar<DT>(void) const;

INSTANTIATE(loco::DataType::S64);
INSTANTIATE(loco::DataType::S32);
INSTANTIATE(loco::DataType::S16);
INSTANTIATE(loco::DataType::S8);
INSTANTIATE(loco::DataType::FLOAT32);
INSTANTIATE(loco::DataType::U8);
INSTANTIATE(loco::DataType::BOOL);

#undef INSTANTIATE

} // namespace luci
