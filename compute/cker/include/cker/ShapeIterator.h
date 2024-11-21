/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __NNFW_CKER_SHAPE_ITERATOR_H__
#define __NNFW_CKER_SHAPE_ITERATOR_H__

#include <utility>
#include "cker/Shape.h"

namespace nnfw
{
namespace cker
{
struct ShapeIterator
{
  /// Definition of this iterator's traits that can be accessed by std::iterator_traits<It>
  using value_type = decltype(std::declval<Shape>().Dims(0));
  using difference_type = std::ptrdiff_t;
  using pointer = value_type *;
  using reference = value_type &;
  using iterator_category = std::bidirectional_iterator_tag;

  ShapeIterator(const Shape &s) : _shape{s}, _current{0}, _last{s.DimensionsCount()} {}
  static ShapeIterator end_iterator(const Shape &s) { return ShapeIterator(s, EndIteratorTag{}); }

  ShapeIterator &operator++()
  {
    ++_current;
    return *this;
  }

  // postincrement
  ShapeIterator operator++(int)
  {
    auto copy = *this;
    ++_current;
    return copy;
  }

  ShapeIterator &operator--()
  {
    --_current;
    return *this;
  }

  ShapeIterator operator--(int)
  {
    auto copy = *this;
    --_current;
    return copy;
  }

  bool operator!=(const ShapeIterator &other) const { return _current != other._current; }
  bool operator==(const ShapeIterator &other) const { return _current == other._current; }

  /// Because the underlying method returns by-value, this operator does the same
  /// instead of returning by-reference like most iterators do.
  value_type operator*() const { return _shape.Dims(_current); }

private:
  struct EndIteratorTag
  {
  };
  // Creates an iterator instance pointing to the past-the-end element
  // This iterator doesn't point to a valid element and thus its dereference is undefined behavior
  ShapeIterator(const Shape &s, EndIteratorTag)
    : _shape{s}, _current{s.DimensionsCount()}, _last{s.DimensionsCount()}
  {
  }

  const Shape &_shape;
  int32_t _current = 0, _last = 0;
};

inline ShapeIterator begin(const Shape &s) { return ShapeIterator(s); }
inline ShapeIterator end(const Shape &s) { return ShapeIterator::end_iterator(s); }

} // namespace cker
} // namespace nnfw

#endif //
