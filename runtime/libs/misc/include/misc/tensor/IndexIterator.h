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
 * @file IndexIterator.h
 * @ingroup COM_AI_RUNTIME
 * @brief This file contains nnfw::misc::tensor::IndexIterator class and
 *        helper function and operator
 */
#ifndef __NNFW_MISC_TENSOR_INDEX_ITERATOR_H__
#define __NNFW_MISC_TENSOR_INDEX_ITERATOR_H__

#include "misc/tensor/Shape.h"
#include "misc/tensor/Index.h"
#include "misc/tensor/IndexEnumerator.h"

namespace nnfw
{
namespace misc
{
namespace tensor
{

/**
 * @brief Class to iterate indexes available for given shape
 */
class IndexIterator
{
public:
  /**
   * @brief Construct a new @c IndexIterator object
   * @param[in] shape   Shape of tensor of which index will be iterated
   */
  IndexIterator(const Shape &shape) : _shape(shape)
  {
    // DO NOTHING
  }

public:
  /**
   * @brief Construct a new IndexIterator object using reference
   * @param[in] IndexIterator    @c IndexIterator object to move
   */
  IndexIterator(IndexIterator &&) = default;

  /**
   * @brief Prevent copy constructor
   */
  IndexIterator(const IndexIterator &) = delete;

public:
  /**
   * @brief Iterate all available indexes and run a function for each index
   * @param[in] fn      Function that requires an index as a parameter.
   * @return @c IndexIterator object
   */
  template <typename Callable> IndexIterator &iter(Callable fn)
  {
    for (IndexEnumerator e{_shape}; e.valid(); e.advance())
    {
      fn(e.curr());
    }

    return (*this);
  }

private:
  const Shape &_shape;
};

/**
 * @brief Get an @c IndexItator object
 * @param[in] shape     Shape of tensor of which index will be iterated
 * @return @c IndexIterator object
 */
inline IndexIterator iterate(const Shape &shape) { return IndexIterator{shape}; }

/**
 * @brief Iterate all indexes and apply a function
 * @param[in] it    @c IndexIterator object that is constructed with a tensor shape
 * @param[in] cb    A function that will receive a specific index.
 *                  Inside the function, the index is used to manipulate tensor element.
 * @return @c IndexIterator object
 */
template <typename Callable> IndexIterator &operator<<(IndexIterator &&it, Callable cb)
{
  return it.iter(cb);
}

} // namespace tensor
} // namespace misc
} // namespace nnfw

#endif // __NNFW_MISC_TENSOR_INDEX_ITERATOR_H__
