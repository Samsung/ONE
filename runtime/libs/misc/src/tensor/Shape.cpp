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

#include "misc/tensor/Shape.h"

#include <cassert>
#include <functional>
#include <numeric>

namespace nnfw
{
namespace misc
{
namespace tensor
{

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

Shape Shape::from(const std::string &str)
{
  Shape shape(0);

  bool pending = false;
  int value = 0;

  for (const char *cur = str.c_str(); true; ++cur)
  {
    if (*cur == ',' || *cur == '\0')
    {
      if (pending)
      {
        shape.append(value);
      }

      if (*cur == '\0')
      {
        break;
      }

      pending = false;
      value = 0;
      continue;
    }

    assert(*cur >= '0' && *cur <= '9');

    pending = true;
    value *= 10;
    value += *cur - '0';
  }

  return shape;
}

uint64_t Shape::num_elements() const
{
  return std::accumulate(_dimensions.cbegin(), _dimensions.cend(), UINT64_C(1),
                         std::multiplies<uint64_t>());
}

std::ostream &operator<<(std::ostream &os, const Shape &shape)
{
  if (shape.rank() > 0)
  {
    os << shape.dim(0);

    for (uint32_t axis = 1; axis < shape.rank(); ++axis)
    {
      os << "," << shape.dim(axis);
    }
  }

  return os;
}

} // namespace tensor
} // namespace misc
} // namespace nnfw
