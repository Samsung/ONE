/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _NDARRAY_SHAPE_H_
#define _NDARRAY_SHAPE_H_

#include "Common.h"

#include <array>
#include <cassert>
#include <cstddef>

namespace ndarray
{

class Shape
{
public:
  //_dims{} here and later since array does not have std::initializer_list ctor
  // and aggregate initialization is not allowed here
  explicit Shape(size_t rank) noexcept : _dims{}, _rank(rank)
  {
    std::fill(_dims.begin(), _dims.end(), 0);
  }

  Shape(std::initializer_list<size_t> list) noexcept : _dims{}, _rank(list.size())
  {
    std::copy(list.begin(), list.end(), _dims.begin());
  }

  size_t dim(int i) const noexcept { return _dims.at(i); }

  size_t &dim(int i) noexcept { return _dims.at(i); }

  size_t element_count() const noexcept
  {
    uint32_t res = 1;
    for (size_t i = 0; i < rank(); ++i)
      res *= dim(i);
    assert(res <= 0xffffffff);
    return res;
  }

  size_t rank() const noexcept { return _rank; }

private:
  std::array<size_t, NDARRAY_MAX_DIMENSION_COUNT> _dims;
  size_t _rank;
};

} // namespace ndarray

#endif //_NDARRAY_SHAPE_H_
