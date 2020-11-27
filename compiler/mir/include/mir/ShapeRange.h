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

#ifndef _MIR_SHAPE_RANGE_H_
#define _MIR_SHAPE_RANGE_H_

#include <cassert>

#include "mir/Shape.h"
#include "mir/Index.h"

namespace mir
{

class ShapeIter
  : public std::iterator<std::forward_iterator_tag, Index, std::size_t, Index *, Index &>
{
public:
  ShapeIter &operator++()
  {
    if (_shape.rank() > 0)
    {
      auto *pidx = &_index.at(0);
      auto *pshape = &_shape.dim(0);
      int32_t rank = _shape.rank();
      int32_t c = rank - 1;
      pidx[c]++;
      while (pidx[c] >= pshape[c] && c > 0)
      {
        pidx[c] = 0;
        pidx[--c]++;
      }
    }
    _pos++;
    return *this;
  }

  const ShapeIter operator++(int)
  {
    ShapeIter it = *this;
    ++*this;
    return it;
  }

  const Index &operator*() const { return _index; }

  bool operator!=(const ShapeIter &iter) const
  {
    assert(iter._index.rank() == _index.rank());
    assert(iter._shape == _shape);
    return _pos != iter._pos;
  }

private:
  explicit ShapeIter(Shape &shape, int32_t pos) : _pos(pos), _shape(shape)
  {
    _index.resize(shape.rank());
    _index.fill(0);
  }

  int32_t _pos;
  Index _index;
  Shape &_shape;

  friend class ShapeRange;
};

class ShapeRange
{
public:
  explicit ShapeRange(const Shape &shape) : _shape(shape) {}

  explicit ShapeRange(Shape &&shape) : _shape(std::move(shape)) {}

  ShapeIter begin() { return ShapeIter(_shape, 0); }

  ShapeIter end() { return ShapeIter(_shape, _shape.numElements()); }

  bool contains(const Index &idx)
  {
    assert(idx.rank() == _shape.rank());
    for (int32_t d = 0; d < idx.rank(); ++d)
    {
      if ((idx.at(d) >= _shape.dim(d)) || (idx.at(d) < 0))
        return false;
    }
    return true;
  }

private:
  Shape _shape;
};

} // namespace mir

#endif //_MIR_SHAPE_RANGE_H_
