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

#include "nncc/core/ADT/tensor/IndexEnumerator.h"

#include <cassert>

using nncc::core::ADT::tensor::Shape;

inline uint32_t axis_of(const Shape &shape, uint32_t cursor)
{
  const uint32_t rank = shape.rank();
  assert(cursor < rank);
  return rank - cursor - 1;
}

namespace nncc
{
namespace core
{
namespace ADT
{
namespace tensor
{

IndexEnumerator::IndexEnumerator(const Shape &shape) : _shape{shape}, _cursor(0)
{
  const uint32_t rank = _shape.rank();

  // Initialize _index
  _index.resize(rank);
  for (uint32_t axis = 0; axis < rank; ++axis)
  {
    _index.at(axis) = 0;
  }

  // Initialize _cursor
  for (_cursor = 0; _cursor < rank; ++_cursor)
  {
    const auto axis = axis_of(_shape, _cursor);

    if (_index.at(axis) < _shape.dim(axis))
    {
      break;
    }
  }
}

void IndexEnumerator::advance(void)
{
  const uint32_t rank = _shape.rank();

  // Find axis to be updated
  while (_cursor < rank)
  {
    const auto axis = axis_of(_shape, _cursor);

    if ((_index.at(axis)) + 1 < _shape.dim(axis))
    {
      break;
    }

    ++_cursor;
  }

  if (_cursor == rank)
  {
    return;
  }

  // Update index
  _index.at(axis_of(_shape, _cursor)) += 1;

  for (uint32_t pos = 0; pos < _cursor; ++pos)
  {
    const auto axis = axis_of(_shape, pos);
    _index.at(axis) = 0;
  }

  // Reset cursor
  _cursor = 0;
}

} // namespace tensor
} // namespace ADT
} // namespace core
} // namespace nncc
