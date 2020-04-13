/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "TFLNodes.h"

#include "Check.h"

#include <loco.h>

#include <cassert>

namespace locoex
{

template <loco::DataType DT> uint32_t TFLConst::size(void) const
{
  assert(dtype() == DT);
  assert(_data.size() % sizeof(typename loco::DataTypeImpl<DT>::Type) == 0);
  return _data.size() / sizeof(typename loco::DataTypeImpl<DT>::Type);
}

template <loco::DataType DT> void TFLConst::size(uint32_t l)
{
  assert(dtype() == DT);
  _data.resize(l * sizeof(typename loco::DataTypeImpl<DT>::Type));
}

template <loco::DataType DT>
const typename loco::DataTypeImpl<DT>::Type &TFLConst::at(uint32_t n) const
{
  assert(dtype() == DT);
  assert(n < size<DT>());
  return *(reinterpret_cast<const typename loco::DataTypeImpl<DT>::Type *>(_data.data()) + n);
}

template <loco::DataType DT> typename loco::DataTypeImpl<DT>::Type &TFLConst::at(uint32_t n)
{
  assert(dtype() == DT);
  assert(n < size<DT>());
  return *(reinterpret_cast<typename loco::DataTypeImpl<DT>::Type *>(_data.data()) + n);
}

#define INSTANTIATE(DT)                                                                   \
  template uint32_t TFLConst::size<DT>(void) const;                                       \
  template void TFLConst::size<DT>(uint32_t);                                             \
  template const typename loco::DataTypeImpl<DT>::Type &TFLConst::at<DT>(uint32_t) const; \
  template typename loco::DataTypeImpl<DT>::Type &TFLConst::at<DT>(uint32_t);

INSTANTIATE(loco::DataType::S32);
INSTANTIATE(loco::DataType::FLOAT32);

#undef INSTANTIATE

void set_new_shape(locoex::TFLReshape *node, int32_t *base, uint32_t size)
{
  // Check node does not have both of new shape infos
  EXO_ASSERT(node->shape() == nullptr, "node already has shape input");
  EXO_ASSERT(node->newShape()->rank() == 0, "node already has newShape attribute");

  const loco::DataType S32 = loco::DataType::S32;

  // Set 2nd input as TFLConst
  auto const_shape_node = node->graph()->nodes()->create<locoex::TFLConst>();
  const_shape_node->rank(1);
  const_shape_node->dim(0) = size;
  const_shape_node->dtype(S32);
  const_shape_node->size<S32>(size);
  for (uint32_t axis = 0; axis < size; ++axis)
    const_shape_node->at<S32>(axis) = base[axis];
  node->shape(const_shape_node);

  // Set newShape attribute
  node->newShape()->rank(size);
  for (uint32_t axis = 0; axis < size; ++axis)
    node->newShape()->dim(axis) = base[axis];
}

} // namespace locoex
