/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Tensor.h"

#include <luci/IR/CircleNodeDecl.h>

#include <cassert>

namespace
{

// Return number of elements of the node.
uint32_t numElements(const luci::CircleNode *node)
{
  uint32_t num_elem = 1;
  for (uint32_t i = 0; i < node->rank(); ++i)
    num_elem *= node->dim(i).value();
  return num_elem;
}

} // namespace

namespace circle_eval_diff
{

#define THROW_UNLESS(COND, MSG) \
  if (not(COND))                \
    throw std::runtime_error(MSG);

template <loco::DataType DT> uint32_t Tensor::size(void) const
{
  assert(dtype() == DT);
  assert(_data.size() % sizeof(typename loco::DataTypeImpl<DT>::Type) == 0);
  return _data.size() / sizeof(typename loco::DataTypeImpl<DT>::Type);
}

template <loco::DataType DT> void Tensor::size(uint32_t l)
{
  assert(dtype() == DT);
  _data.resize(l * sizeof(typename loco::DataTypeImpl<DT>::Type));
}

template <loco::DataType DT>
const typename loco::DataTypeImpl<DT>::Type &Tensor::at(uint32_t n) const
{
  assert(dtype() == DT);
  THROW_UNLESS(n < size<DT>(), "Access to out of buffer boundary.");
  return *(reinterpret_cast<const typename loco::DataTypeImpl<DT>::Type *>(_data.data()) + n);
}

template <loco::DataType DT> typename loco::DataTypeImpl<DT>::Type &Tensor::at(uint32_t n)
{
  assert(dtype() == DT);
  THROW_UNLESS(n < size<DT>(), "Access to out of buffer boundary.");
  return *(reinterpret_cast<typename loco::DataTypeImpl<DT>::Type *>(_data.data()) + n);
}

#undef THROW_UNLESS

#define INSTANTIATE(DT)                                                                 \
  template uint32_t Tensor::size<DT>(void) const;                                       \
  template void Tensor::size<DT>(uint32_t);                                             \
  template const typename loco::DataTypeImpl<DT>::Type &Tensor::at<DT>(uint32_t) const; \
  template typename loco::DataTypeImpl<DT>::Type &Tensor::at<DT>(uint32_t);

INSTANTIATE(loco::DataType::S64);
INSTANTIATE(loco::DataType::S32);
INSTANTIATE(loco::DataType::S16);
INSTANTIATE(loco::DataType::U8);
INSTANTIATE(loco::DataType::FLOAT32);

#undef INSTANTIATE

// Return Tensor which has the same dtype and shape with node.
// Buffer does not have any data yet.
std::shared_ptr<Tensor> createEmptyTensor(const luci::CircleNode *node)
{
  auto tensor = std::make_shared<Tensor>();
  {
    tensor->dtype(node->dtype());
    tensor->rank(node->rank());
    for (uint32_t i = 0; i < node->rank(); i++)
      tensor->dim(i) = node->dim(i);

    switch (node->dtype())
    {
      case loco::DataType::FLOAT32:
        tensor->size<loco::DataType::FLOAT32>(numElements(node));
        break;
      case loco::DataType::U8:
        tensor->size<loco::DataType::U8>(numElements(node));
        break;
      case loco::DataType::S16:
        tensor->size<loco::DataType::S16>(numElements(node));
        break;
      case loco::DataType::S32:
        tensor->size<loco::DataType::S32>(numElements(node));
        break;
      case loco::DataType::S64:
        tensor->size<loco::DataType::S64>(numElements(node));
        break;
      default:
        throw std::runtime_error("Unsupported input tensor dtype for " + node->name());
    }
  }

  return tensor;
}

} // namespace circle_eval_diff
