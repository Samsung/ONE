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
 * @file Zipper.h
 * @ingroup COM_AI_RUNTIME
 * @brief This file contains nnfw::misc::tensor::Zipper class
 */

#ifndef __NNFW_MISC_TENSOR_ZIPPER_H__
#define __NNFW_MISC_TENSOR_ZIPPER_H__

#include "misc/tensor/Index.h"
#include "misc/tensor/IndexIterator.h"
#include "misc/tensor/Reader.h"

namespace nnfw
{
namespace misc
{
namespace tensor
{

/**
 * @brief Class to apply a function with three params: @c Index, elements of a tensor
 * at passed index read by @c Reader objects
 */
template <typename T> class Zipper
{
public:
  /**
   * @brief Construct a new @c Zipper object
   * @param[in] shape   Shape of @c lhs and @c rhs
   * @param[in] lhs     @c Reader object of a tensor
   * @param[in] rhs     @c Reader object of a tensor
   */
  Zipper(const Shape &shape, const Reader<T> &lhs, const Reader<T> &rhs)
    : _shape{shape}, _lhs{lhs}, _rhs{rhs}
  {
    // DO NOTHING
  }

public:
  /**
   * @brief Apply @c cb to all elements of tensors. Elements of two tensors
   *        at passed @c index are read by @c lhs and @c rhs
   * @param[in] cb   Function to apply
   * @return    N/A
   */
  template <typename Callable> void zip(Callable cb) const
  {
    iterate(_shape) <<
      [this, &cb](const Index &index) { cb(index, _lhs.at(index), _rhs.at(index)); };
  }

private:
  const Shape &_shape;
  const Reader<T> &_lhs;
  const Reader<T> &_rhs;
};

/**
 * @brief Apply @c cb by using @c lhs and @c rhs passed to the constructor of @c zipper
 * @param[in] zipper    @c Zipper object
 * @param[in] cb        Function to zpply using @c zip function
 * @return @c zipper object after applying @c cb to @c zipper
 */
template <typename T, typename Callable>
const Zipper<T> &operator<<(const Zipper<T> &zipper, Callable cb)
{
  zipper.zip(cb);
  return zipper;
}

/**
 * @brief Get @c Zipper object constructed using passed params
 * @param shape   Shape of @c lhs and @c rhs
 * @param lhs     @c Reader object of a tensor
 * @param rhs     @c Reader object of a tensor
 * @return        @c Zipper object
 */
template <typename T> Zipper<T> zip(const Shape &shape, const Reader<T> &lhs, const Reader<T> &rhs)
{
  return Zipper<T>{shape, lhs, rhs};
}

} // namespace tensor
} // namespace misc
} // namespace nnfw

#endif // __NNFW_MISC_TENSOR_ZIPPER_H__
