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

#ifndef _MIR_SHAPE_H_
#define _MIR_SHAPE_H_

#include <initializer_list>
#include <vector>
#include <cstdint>

#include "adtidas/SmallVector.h"
#include "mir/Common.h"

namespace mir
{

class Shape
{
public:
  static constexpr int32_t autoDim = -1;

  Shape() = default;

  explicit Shape(int32_t rank) : _dims(rank) {}

  Shape(std::initializer_list<int32_t> &&dims) : _dims(std::move(dims)) {}

  explicit Shape(const std::vector<int32_t> &dims) : _dims(std::begin(dims), std::end(dims)) {}

  int32_t rank() const { return static_cast<int32_t>(_dims.size()); }

  void resize(int32_t size);

  int32_t &dim(int32_t axis) noexcept
  {
    auto dim = wrap_index(axis, _dims.size());
    return _dims[dim];
  };

  int32_t dim(int32_t axis) const noexcept
  {
    auto dim = wrap_index(axis, _dims.size());
    return _dims[dim];
  }

  int32_t numElements() const;

  bool operator==(const Shape &rhs) const { return _dims == rhs._dims; }

  bool operator!=(const Shape &rhs) const { return !(*this == rhs); }

private:
  adt::small_vector<int32_t, MAX_DIMENSION_COUNT> _dims;
};

Shape broadcastShapes(const Shape &lhs_shape, const Shape &rhs_shape);

std::string toString(const Shape &shape);

} // namespace mir

#endif //_MIR_SHAPE_H_
