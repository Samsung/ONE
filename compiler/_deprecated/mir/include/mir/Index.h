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

#ifndef _MIR_INDEX_H_
#define _MIR_INDEX_H_

#include <initializer_list>
#include <cstdint>
#include <ostream>

#include "mir/Common.h"

#include "adtidas/SmallVector.h"

namespace mir
{

class Index
{
public:
  Index() = default;

  Index(std::initializer_list<int32_t> &&l) noexcept : _indices(std::move(l))
  {
    // DO NOTHING
  }

  explicit Index(size_t rank) : _indices(rank) {}

  int32_t rank() const { return static_cast<int32_t>(_indices.size()); }

  /**
   * @brief resize index to given dimension number
   * @param size new number of dimensions
   * @return *this
   * @warning if new size is greater than old, new dimensions are undefined
   */
  Index &resize(int32_t size);

  /**
   * @brief fill all axis with `index`
   * @return `*this`
   */
  Index &fill(int32_t index);

  /**
   * @brief return position on given axis
   * @param axis index of axis to get index on. If axis is negative returns axis from the last
   * @return
   */
  int32_t &at(int32_t axis) { return _indices[wrap_index(axis, _indices.size())]; }

  /**
   * @brief return position on given axis
   * @param axis index of axis to get index on. If axis is negative returns axis from the last
   * @return
   */
  int32_t at(int32_t axis) const { return _indices[wrap_index(axis, _indices.size())]; }

private:
  adt::small_vector<int32_t, MAX_DIMENSION_COUNT> _indices;
};

std::ostream &operator<<(std::ostream &s, const Index &idx);

} // namespace mir

#endif //_MIR_INDEX_H_
