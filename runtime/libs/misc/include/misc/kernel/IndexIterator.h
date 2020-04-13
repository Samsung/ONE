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
 * @file     IndexIterator.h
 * @brief    This file contains IndexIterator class
 * @ingroup  COM_AI_RUNTIME
 */

#ifndef __NNFW_MISC_KERNEL_INDEX_ITERATOR_H__
#define __NNFW_MISC_KERNEL_INDEX_ITERATOR_H__

#include "misc/kernel/Shape.h"

namespace nnfw
{
namespace misc
{
namespace kernel
{

/**
 * @brief Class to iterate Callable with Index of kernel
 */
class IndexIterator
{
public:
  /**
   * @brief Construct IndexIterator object with Shape of kernel
   * @param[in] shape Shape reference of feature
   */
  IndexIterator(const Shape &shape) : _shape{shape}
  {
    // DO NOTHING
  }

public:
  /**
   * @brief Call a function iterated
   * @param[in] cb A callback function
   * @return Current IndexIterator object
   */
  template <typename Callable> IndexIterator &iter(Callable cb)
  {
    for (int32_t nth = 0; nth < _shape.N; ++nth)
    {
      for (int32_t ch = 0; ch < _shape.C; ++ch)
      {
        for (int32_t row = 0; row < _shape.H; ++row)
        {
          for (int32_t col = 0; col < _shape.W; ++col)
          {
            cb(nth, ch, row, col);
          }
        }
      }
    }

    return (*this);
  }

private:
  const Shape _shape; /**< Shape for kernel */
};

/**
 * @brief Create an object of IndexIterator for kernel
 * @param[in] shape reference of feature
 * @return Created IndexIterator object
 */
inline IndexIterator iterate(const Shape &shape) { return IndexIterator{shape}; }

/**
 * @brief Call a function iterated using IndexIterator of kernel
 *        Overloaded operator<<
 * @param[in] it An IndexIterator reference
 * @param[in] cb A callback function
 * @return Created IndexIterator object
 */
template <typename Callable> IndexIterator &operator<<(IndexIterator &&it, Callable cb)
{
  return it.iter(cb);
}

} // namespace kernel
} // namespace misc
} // namespace nnfw

#endif // __NNFW_MISC_FEATURE_INDEX_ITERATOR_H__
