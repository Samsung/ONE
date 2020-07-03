/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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
 */

#ifndef __ONERT_EXEC_FEATURE_INDEX_ITERATOR_H__
#define __ONERT_EXEC_FEATURE_INDEX_ITERATOR_H__

#include "ir/Shape.h"

namespace onert
{
namespace exec
{
namespace feature
{

/**
 * @brief  Class to iterate Callable with Index of feature
 */
class IndexIterator
{
public:
  /**
   * @brief     Construct IndexIterator object with Shape of feature
   * @param[in] shape Shape reference of feature
   */
  IndexIterator(const ir::FeatureShape &shape) : _shape{shape}
  {
    // DO NOTHING
  }

public:
  /**
   * @brief     Call a function iterated
   * @param[in] cb  A callback function
   * @return    Current IndexIterator object
   */
  template <typename Callable> IndexIterator &iter(Callable cb)
  {
    for (int32_t batch = 0; batch < _shape.N; ++batch)
    {
      for (int32_t ch = 0; ch < _shape.C; ++ch)
      {
        for (int32_t row = 0; row < _shape.H; ++row)
        {
          for (int32_t col = 0; col < _shape.W; ++col)
          {
            cb(batch, ch, row, col);
          }
        }
      }
    }

    return (*this);
  }

private:
  /**
   * @brief Shape for feature
   */
  const ir::FeatureShape _shape;
};

/**
 * @brief     Create an object of IndexIterator for feature
 * @param[in] Shape reference of feature
 * @return    Created IndexIterator object
 */
static inline IndexIterator iterate(const ir::FeatureShape &shape) { return IndexIterator{shape}; }

/**
 * @brief     Call a function iterated using IndexIterator of feature
 *            Overloaded operator<<
 * @param[in] it  An IndexIterator reference
 * @param[in] cb  A callback function
 * @return    created IndexIterator object
 */
template <typename Callable> IndexIterator &operator<<(IndexIterator &&it, Callable cb)
{
  return it.iter(cb);
}

} // namespace feature
} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_FEATURE_INDEX_ITERATOR_H__
