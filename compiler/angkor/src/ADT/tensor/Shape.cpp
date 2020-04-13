/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "nncc/core/ADT/tensor/Shape.h"

#include <algorithm>

namespace nncc
{
namespace core
{
namespace ADT
{
namespace tensor
{

Shape::Shape(std::initializer_list<uint32_t> &&l) : _dims{l}
{
  // DO NOTHING
}

uint32_t Shape::rank(void) const { return _dims.size(); }
Shape &Shape::resize(uint32_t size)
{
  _dims.resize(size);
  return *this;
}

uint32_t &Shape::dim(uint32_t axis) { return _dims.at(axis); }
uint32_t Shape::dim(uint32_t axis) const { return _dims.at(axis); }

Shape &Shape::squeeze(void)
{
  _dims.erase(std::remove(_dims.begin(), _dims.end(), 1), _dims.end());
  return *this;
}

uint64_t num_elements(const Shape &shape)
{
  uint64_t res = 1;

  for (uint32_t axis = 0; axis < shape.rank(); ++axis)
  {
    res *= shape.dim(axis);
  }

  return res;
}

Shape squeeze(const Shape &shape)
{
  Shape res{shape};
  res.squeeze();
  return res;
}

bool operator==(const Shape &lhs, const Shape &rhs)
{
  if (lhs.rank() != rhs.rank())
  {
    return false;
  }

  for (uint32_t axis = 0; axis < lhs.rank(); ++axis)
  {
    if (lhs.dim(axis) != rhs.dim(axis))
    {
      return false;
    }
  }

  return true;
}

} // namespace tensor
} // namespace ADT
} // namespace core
} // namespace nncc
