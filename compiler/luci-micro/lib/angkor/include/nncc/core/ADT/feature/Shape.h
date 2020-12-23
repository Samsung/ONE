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

#ifndef __NNCC_CORE_ADT_FEATURE_SHAPE_H__
#define __NNCC_CORE_ADT_FEATURE_SHAPE_H__

#include <cstdint>

namespace nncc
{
namespace core
{
namespace ADT
{
namespace feature
{

//
// Shape of Feature Map for Convolution
//
class Shape
{
public:
  Shape(uint32_t depth, uint32_t height, uint32_t width)
      : _depth{depth}, _height{height}, _width{width}
  {
    // DO NOTHING
  }

public:
  uint32_t depth(void) const { return _depth; }
  uint32_t height(void) const { return _height; }
  uint32_t width(void) const { return _width; }

private:
  uint32_t _depth;
  uint32_t _height;
  uint32_t _width;
};

/**
 * @brief The number of elements of a feature map of a given shape
 *
 * WARN The result is valid only when the expected value is less than 2^32 - 1
 */
inline uint32_t num_elements(const Shape &shape)
{
  return shape.depth() * shape.height() * shape.width();
}

inline bool operator==(const Shape &l, const Shape &r)
{
  return (l.depth() == r.depth()) && (l.height() == r.height()) && (l.width() == r.width());
}

} // namespace feature
} // namespace ADT
} // namespace core
} // namespace nncc

#endif // __NNCC_CORE_ADT_FEATURE_SHAPE_H__
