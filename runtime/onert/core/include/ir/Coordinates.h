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

#ifndef __ONERT_IR_COORDINATES_H__
#define __ONERT_IR_COORDINATES_H__

#include <cassert>
#include <stdint.h>
#include <vector>

#include "Layout.h"

namespace onert
{
namespace ir
{

/**
 * @brief Class to represent position(offset) of tensor.\n
 *        Assume that the front is higher dimensional.
 *        i.g. N: 0, C: 1, H: 2, W: 3 for NCHW layout
 */
class Coordinates final
{
public:
  static constexpr size_t num_max_dimensions = 4;

public:
  /**
   * @brief     Construct a new Coordinates object with zero dimension
   * @return    N/A
   */
  Coordinates() = default;
  /**
   * @brief     Construct a new Coordinates object
   * @param[in] init The initialzer_list with coordinates
   * @return
   */
  Coordinates(std::initializer_list<int32_t> init) : _coordinates{init}
  {
    assert(init.size() <= num_max_dimensions);
  }
  /**
   * @brief     Construct a new Coordinates object
   * @param[in] init The initialzer_list with coordinates
   * @return
   */
  Coordinates(std::initializer_list<uint32_t> init) : _coordinates{init.begin(), init.end()}
  {
    assert(init.size() <= num_max_dimensions);
  }

public:
  /**
   * @brief  Set the coordinate of one of the coordinates.
   *
   * @param[in] dimension  Dimension for which the coordinate is set.
   * @param[in] Coordinate Coordinate to be set for the dimension.
   */
  void set(size_t dimension, int32_t coordinate)
  {
    assert(dimension < num_max_dimensions);
    if (dimension >= _coordinates.size())
    {
      _coordinates.resize(dimension + 1, 0);
    }
    _coordinates[dimension] = coordinate;
  }

public:
  /**
   * @brief Return size of coordinates
   *
   * @return size of coordinates
   */
  size_t size() const { return _coordinates.size(); }

public:
  int32_t operator[](size_t dimension) const
  {
    assert(dimension < _coordinates.size());
    return _coordinates[dimension];
  }

public:
  /**
   * @brief begin() of const_iterator for this class
   *
   * @return The first iterator of the coordinates
   */
  std::vector<int32_t>::const_iterator begin() const { return _coordinates.begin(); }
  /**
   * @brief end() of const_iterator for this class
   *
   * @return The last iterator of the coordinates
   */
  std::vector<int32_t>::const_iterator end() const { return _coordinates.end(); }

private:
  std::vector<int32_t> _coordinates;
};

Coordinates convertCoordinates(const Coordinates &from_coordinates, Layout from_layout,
                               Layout to_layout);

} // namespace ir
} // namespace onert

#endif // __ONERT_IR_COORDINATES_H__
