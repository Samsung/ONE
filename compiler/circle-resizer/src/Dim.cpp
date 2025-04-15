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

#include "Dim.h"

#include <stdexcept>

using namespace circle_resizer;

Dim::Dim(int32_t dim) : _dim_value{dim}
{
  if (dim < -1)
  {
    throw std::runtime_error("Invalid value of dimension: " + dim);
  }
}

Dim Dim::dynamic() { return Dim{-1}; }

bool Dim::is_dynamic() const { return _dim_value == -1; }

int32_t Dim::value() const { return _dim_value; }

bool Dim::operator==(const Dim &rhs) const { return value() == rhs.value(); }
