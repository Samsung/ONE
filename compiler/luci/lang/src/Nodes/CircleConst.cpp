/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/IR/Nodes/CircleConst.h"

#include <cassert>

namespace luci
{

template <loco::DataType DT> uint32_t CircleConst::size(void) const
{
  assert(dtype() == DT);
  assert(_data.size() % sizeof(typename loco::DataTypeImpl<DT>::Type) == 0);
  return _data.size() / sizeof(typename loco::DataTypeImpl<DT>::Type);
}

template <loco::DataType DT> void CircleConst::size(uint32_t l)
{
  assert(dtype() == DT);
  _data.resize(l * sizeof(typename loco::DataTypeImpl<DT>::Type));
}

template <loco::DataType DT>
const typename loco::DataTypeImpl<DT>::Type &CircleConst::at(uint32_t n) const
{
  assert(dtype() == DT);
  assert(n < size<DT>());
  return *(reinterpret_cast<const typename loco::DataTypeImpl<DT>::Type *>(_data.data()) + n);
}

template <loco::DataType DT> typename loco::DataTypeImpl<DT>::Type &CircleConst::at(uint32_t n)
{
  assert(dtype() == DT);
  assert(n < size<DT>());
  return *(reinterpret_cast<typename loco::DataTypeImpl<DT>::Type *>(_data.data()) + n);
}

template <loco::DataType DT>
const typename loco::DataTypeImpl<DT>::Type &CircleConst::scalar(void) const
{
  assert(dtype() == DT);
  return *(reinterpret_cast<const typename loco::DataTypeImpl<DT>::Type *>(_data.data()));
}

template <loco::DataType DT> typename loco::DataTypeImpl<DT>::Type &CircleConst::scalar(void)
{
  assert(dtype() == DT);
  return *(reinterpret_cast<typename loco::DataTypeImpl<DT>::Type *>(_data.data()));
}

#define INSTANTIATE(DT)                                                                      \
  template uint32_t CircleConst::size<DT>(void) const;                                       \
  template void CircleConst::size<DT>(uint32_t);                                             \
  template const typename loco::DataTypeImpl<DT>::Type &CircleConst::at<DT>(uint32_t) const; \
  template typename loco::DataTypeImpl<DT>::Type &CircleConst::at<DT>(uint32_t);             \
  template const typename loco::DataTypeImpl<DT>::Type &CircleConst::scalar<DT>(void) const; \
  template typename loco::DataTypeImpl<DT>::Type &CircleConst::scalar<DT>(void);

INSTANTIATE(loco::DataType::S64);
INSTANTIATE(loco::DataType::S32);
INSTANTIATE(loco::DataType::S16);
INSTANTIATE(loco::DataType::S8);
INSTANTIATE(loco::DataType::S4);
INSTANTIATE(loco::DataType::FLOAT32);
INSTANTIATE(loco::DataType::U8);
INSTANTIATE(loco::DataType::U4);
INSTANTIATE(loco::DataType::BOOL);
INSTANTIATE(loco::DataType::FLOAT16);

#undef INSTANTIATE

// CircleConst implementations for loco::DataType::STRING

template <> uint32_t CircleConst::size<loco::DataType::STRING>(void) const
{
  assert(dtype() == loco::DataType::STRING);
  assert(_data.size() == 0);
  return _strings.size();
}

template <> void CircleConst::size<loco::DataType::STRING>(uint32_t l)
{
  assert(dtype() == loco::DataType::STRING);
  assert(_data.size() == 0);
  _strings.resize(l);
}

template <> const std::string &CircleConst::at<loco::DataType::STRING>(uint32_t n) const
{
  assert(dtype() == loco::DataType::STRING);
  assert(n < _strings.size());
  return _strings.at(n);
}

template <> std::string &CircleConst::at<loco::DataType::STRING>(uint32_t n)
{
  assert(dtype() == loco::DataType::STRING);
  assert(n < _strings.size());
  return _strings.at(n);
}

template <> const std::string &CircleConst::scalar<loco::DataType::STRING>(void) const
{
  assert(dtype() == loco::DataType::STRING);
  assert(1 == _strings.size());
  return _strings.at(0);
}

template <> std::string &CircleConst::scalar<loco::DataType::STRING>(void)
{
  assert(dtype() == loco::DataType::STRING);
  assert(1 == _strings.size());
  return _strings.at(0);
}

} // namespace luci
