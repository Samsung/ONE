/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Shape.h"

#include <algorithm>
#include <limits>

using namespace circle_resizer;

Shape::Shape(const std::initializer_list<Dim> &dims) : _dims{dims} {}

Shape::Shape(const std::vector<Dim> &shape_vec) : _dims{shape_vec} {}

Shape::Shape(const std::initializer_list<uint32_t> &shape_vec)
{
  for (const auto &dim : shape_vec)
  {
    if (dim >= std::numeric_limits<int32_t>::max())
    {
      std::out_of_range("Provided dimension: " + std::to_string(dim) + " is out of range");
    }
    _dims.emplace_back(Dim{static_cast<int32_t>(dim)});
  }
}

Shape Shape::scalar() { return Shape{std::initializer_list<Dim>{}}; }

size_t Shape::rank() const { return _dims.size(); }

Dim Shape::operator[](const size_t &axis) const
{
  if (is_scalar())
  {
    throw std::invalid_argument("You cannot gather dimension from a scalar");
  }
  if (axis > rank() - 1)
  {
    throw std::out_of_range("Axis=" + std::to_string(axis) +
                            " is out of range of shape's rank: " + std::to_string(rank()));
  }
  return _dims[axis];
}

bool Shape::is_scalar() const { return _dims.empty(); }

bool Shape::is_dynamic() const
{
  if (is_scalar())
  {
    return false;
  }
  return std::any_of(std::begin(_dims), std::end(_dims),
                     [](const Dim &dim) { return dim.is_dynamic(); });
}

bool Shape::operator==(const Shape &rhs) const
{
  if (rank() != rhs.rank())
  {
    return false;
  }
  for (size_t axis = 0; axis < rank(); ++axis)
  {
    if (_dims[axis].value() != rhs[axis].value())
    {
      return false;
    }
  }
  return true;
}

std::ostream &circle_resizer::operator<<(std::ostream &os, const Shape &shape)
{
  if (shape.is_scalar())
  {
    os << "[]";
    return os;
  }
  os << "[";
  for (int i = 0; i < shape.rank() - 1; ++i)
  {
    os << shape[i].value() << ", ";
  }
  os << shape[shape.rank() - 1].value() << "]";
  return os;
}
