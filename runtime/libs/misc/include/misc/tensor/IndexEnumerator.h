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

/**
 * @file IndexEnumerator.h
 * @ingroup COM_AI_RUNTIME
 * @brief This file contains nnfw::misc::tensor::IndexEnumerator class
 */

#ifndef __NNFW_MISC_TENSOR_INDEX_ENUMERATOR_H__
#define __NNFW_MISC_TENSOR_INDEX_ENUMERATOR_H__

#include "misc/tensor/Shape.h"
#include "misc/tensor/Index.h"

namespace nnfw
{
namespace misc
{
namespace tensor
{
/**
 * @brief Class to enumerate index of a tensor
 *
 */
class IndexEnumerator
{
public:
  /**
   * @brief Construct a new @c IndexEnumerator object
   * @param[in] shape   Shape of tensor of which index will be enumerate
   */
  explicit IndexEnumerator(const Shape &shape) : _shape(shape), _cursor(0), _index(shape.rank())
  {
    const uint32_t rank = _shape.rank();

    for (uint32_t axis = 0; axis < rank; ++axis)
    {
      _index.at(axis) = 0;
    }

    for (_cursor = 0; _cursor < rank; ++_cursor)
    {
      if (_index.at(_cursor) < _shape.dim(_cursor))
      {
        break;
      }
    }
  }

public:
  /**
   * @brief Prevent constructing @c IndexEnumerator object by using R-value reference
   */
  IndexEnumerator(IndexEnumerator &&) = delete;
  /**
   * @brief Prevent copy constructor
   */
  IndexEnumerator(const IndexEnumerator &) = delete;

public:
  /**
   * @brief Check if more enumeration is available
   * @return @c true if more @c advance() is available, otherwise @c false
   */
  bool valid(void) const { return _cursor < _shape.rank(); }

public:
  /**
   * @brief Get the current index to enumerate
   * @return Current index
   */
  const Index &curr(void) const { return _index; }

public:
  /**
   * @brief Advance index by +1
   */
  void advance(void)
  {
    const uint32_t rank = _shape.rank();

    // Find axis to be updated
    while ((_cursor < rank) && !(_index.at(_cursor) + 1 < _shape.dim(_cursor)))
    {
      ++_cursor;
    }

    if (_cursor == rank)
    {
      return;
    }

    // Update index
    _index.at(_cursor) += 1;

    for (uint32_t axis = 0; axis < _cursor; ++axis)
    {
      _index.at(axis) = 0;
    }

    // Update cursor
    _cursor = 0;
  }

public:
  const Shape _shape; //!< Shape to enumerate

private:
  uint32_t _cursor;
  Index _index;
};

} // namespace tensor
} // namespace misc
} // namespace nnfw

#endif // __NNFW_MISC_TENSOR_INDEX_ENUMERATOR_H__
