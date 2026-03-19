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

#include "moco/IR/Nodes/TFConst.h"

#include <cassert>

namespace moco
{

template <loco::DataType DT> uint32_t TFConst::size(void) const
{
  assert(dtype() == DT);
  assert(_data.size() % sizeof(typename loco::DataTypeImpl<DT>::Type) == 0);
  return _data.size() / sizeof(typename loco::DataTypeImpl<DT>::Type);
}

template <loco::DataType DT> void TFConst::size(uint32_t l)
{
  assert(dtype() == DT);
  _data.resize(l * sizeof(typename loco::DataTypeImpl<DT>::Type));
}

template <loco::DataType DT>
const typename loco::DataTypeImpl<DT>::Type &TFConst::at(uint32_t n) const
{
  assert(dtype() == DT);
  assert(n < size<DT>());
  return *(reinterpret_cast<const typename loco::DataTypeImpl<DT>::Type *>(_data.data()) + n);
}

template <loco::DataType DT> typename loco::DataTypeImpl<DT>::Type &TFConst::at(uint32_t n)
{
  assert(dtype() == DT);
  assert(n < size<DT>());
  return *(reinterpret_cast<typename loco::DataTypeImpl<DT>::Type *>(_data.data()) + n);
}

#define INSTANTIATE(DT)                                                                  \
  template uint32_t TFConst::size<DT>(void) const;                                       \
  template void TFConst::size<DT>(uint32_t);                                             \
  template const typename loco::DataTypeImpl<DT>::Type &TFConst::at<DT>(uint32_t) const; \
  template typename loco::DataTypeImpl<DT>::Type &TFConst::at<DT>(uint32_t);

INSTANTIATE(loco::DataType::S8);
INSTANTIATE(loco::DataType::S32);
INSTANTIATE(loco::DataType::FLOAT32);

#undef INSTANTIATE

loco::TensorShape tensor_shape(const TFConst *node)
{
  assert(node != nullptr);

  loco::TensorShape shape;

  uint32_t rank = node->rank();
  shape.rank(rank);
  for (uint32_t index = 0; index < rank; ++index)
  {
    assert(node->dim(index).known());
    shape.dim(index) = node->dim(index).value();
  }

  return shape;
}

uint32_t num_elements(const TFConst *tfconst)
{
  assert(tfconst != nullptr);

  uint32_t num_elements = 1;
  for (uint32_t index = 0; index < tfconst->rank(); ++index)
  {
    assert(tfconst->dim(index).known());
    uint32_t dim = tfconst->dim(index).value();
    num_elements = num_elements * dim;
  }
  return num_elements;
}

bool same_shape(const TFConst *lhs, const TFConst *rhs)
{
  assert(lhs != nullptr);
  assert(rhs != nullptr);

  if (lhs->rank() != rhs->rank())
    return false;

  for (uint32_t index = 0; index < lhs->rank(); ++index)
  {
    assert(lhs->dim(index).known());
    assert(rhs->dim(index).known());
    if (lhs->dim(index).value() != rhs->dim(index).value())
      return false;
  }
  return true;
}

} // namespace moco
