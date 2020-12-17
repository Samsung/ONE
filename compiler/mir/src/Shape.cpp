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

#include "mir/Shape.h"

#include <algorithm>
#include <cassert>
#include <sstream>

namespace mir
{

constexpr int32_t mir::Shape::autoDim;

void Shape::resize(int32_t size) { _dims.resize(size); }

int32_t Shape::numElements() const
{
  int32_t res = 1;

  for (int32_t axis = 0; axis < rank(); ++axis)
  {
    assert(dim(axis) != Shape::autoDim);
    res *= dim(axis);
  }

  return res;
}

Shape broadcastShapes(const Shape &lhs_shape, const Shape &rhs_shape)
{
  const int num_dims = std::max(lhs_shape.rank(), rhs_shape.rank());
  Shape result_shape(num_dims);

  for (int i = 0; i < num_dims; ++i)
  {
    const std::int32_t lhs_dim =
      (i >= num_dims - lhs_shape.rank()) ? lhs_shape.dim(i - (num_dims - lhs_shape.rank())) : 1;
    const std::int32_t rhs_dim =
      (i >= num_dims - rhs_shape.rank()) ? rhs_shape.dim(i - (num_dims - rhs_shape.rank())) : 1;
    if (lhs_dim == 1)
    {
      result_shape.dim(i) = rhs_dim;
    }
    else
    {
      assert(rhs_dim == 1 || rhs_dim == lhs_dim);
      result_shape.dim(i) = lhs_dim;
    }
  }

  return result_shape;
}

std::string toString(const Shape &shape)
{
  std::stringstream ss;

  ss << "[";
  for (int32_t axis = 0; axis < shape.rank(); ++axis)
  {
    if (axis != 0)
      ss << ", ";
    if (shape.dim(axis) == Shape::autoDim)
      ss << "AUTO";
    else
      ss << shape.dim(axis);
  }
  ss << "]";

  return ss.str();
}

} // namespace mir
