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

#include "nncc/core/ADT/tensor/Index.h"

#include <stdexcept>
#include <algorithm>

namespace nncc
{
namespace core
{
namespace ADT
{
namespace tensor
{

Index::Index(std::initializer_list<uint32_t> &&l) : _indices{l}
{
  // DO NOTHING
}

uint32_t Index::rank(void) const { return _indices.size(); }
Index &Index::resize(uint32_t size)
{
  _indices.resize(size);
  return *this;
}

Index &Index::fill(uint32_t index)
{
  std::fill(_indices.begin(), _indices.end(), index);
  return (*this);
}

uint32_t &Index::at(uint32_t axis) { return _indices.at(axis); }
uint32_t Index::at(uint32_t axis) const { return _indices.at(axis); }

Index operator+(const Index &lhs, const Index &rhs)
{
  if (lhs.rank() != rhs.rank())
    throw std::runtime_error("Two tensors should have same rank");

  Index ret;
  ret.resize(lhs.rank());
  for (uint32_t axis = 0; axis < lhs.rank(); axis++)
  {
    ret.at(axis) = lhs.at(axis) + rhs.at(axis);
  }
  return ret;
}

bool operator==(const Index &lhs, const Index &rhs)
{
  if (lhs.rank() != rhs.rank())
    return false;
  for (uint32_t axis = 0; axis < lhs.rank(); axis++)
  {
    if (lhs.at(axis) != rhs.at(axis))
      return false;
  }
  return true;
}

} // namespace tensor
} // namespace ADT
} // namespace core
} // namespace nncc
