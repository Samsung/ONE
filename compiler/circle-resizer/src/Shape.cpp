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

#include <stdexcept>

using namespace circle_resizer;

Dim::Dim(const std::optional<int32_t> &dim) : _dim{dim} {}

Dim::Dim(int32_t dim_value) : Dim{std::optional<int32_t>{dim_value}}
{
  if (_dim.value() < -1)
  {
    throw std::runtime_error("Invalid value of dimension: " + _dim.value());
  }
}

Dim Dim::scalar() { return Dim{std::nullopt}; }

bool Dim::is_scalar() const { return !_dim.has_value(); }

bool Dim::is_dynamic() const { return _dim.value() == -1; }

int32_t Dim::value() const
{
  if (!_dim.has_value())
  {
    std::runtime_error("The dimension is a scalar");
  }
  return _dim.value();
}

bool Dim::operator==(const Dim &rhs) const
{
  if (is_scalar() && rhs.is_scalar())
  {
    return true;
  }
  if (is_scalar() != rhs.is_scalar())
  {
    return false;
  }
  return value() == rhs.value();
}
