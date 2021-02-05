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
 * @file Comparator.h
 * @ingroup COM_AI_RUNTIME
 * @brief This file contains nnfw::misc::tensor::Comparator class
 */

#ifndef __NNFW_MISC_TENSOR_COMPARATOR_H__
#define __NNFW_MISC_TENSOR_COMPARATOR_H__

#include "misc/tensor/Index.h"
#include "misc/tensor/Shape.h"
#include "misc/tensor/Reader.h"
#include "misc/tensor/Diff.h"

#include <functional>

#include <vector>

namespace nnfw
{
namespace misc
{
namespace tensor
{

/**
 * @brief Class to compare two tensors (expected and obtained to compare)
 */
template <typename T> class Comparator
{
public:
  /**
   * @brief Construct a new @c Comparator object
   * @param[in] fn     Function that compares two T values
   */
  Comparator(const std::function<bool(T lhs, T rhs)> &fn) : _compare_fn{fn}
  {
    // DO NOTHING
  }

public:
  /**
   * @brief Struct to observe comparison results
   */
  struct Observer
  {
    /**
     * @brief Get notification of comparison result at every index of two tensors
     * @param[in] index       Index of tensors compared
     * @param[in] expected    Expected value of element at @c index
     * @param[in] obtained    Obtained value of element at @c index
     * @return    N/A
     */
    virtual void notify(const Index &index, T expected, T obtained) = 0;
  };

public:
  /**
   * @brief Compare two tensors
   * @param[in] shape       Shape of two tensors
   * @param[in] expected    @c Reader<T> object that accesses expected tensor
   * @param[in] obtained    @c Reader<T> object that accesses obtained tensor
   * @param[in] observer    @c Observer notified of expected value and obtained value at every index
   * @return    @c std::vector<Diff<T>> containing information of failed comparison
   */
  // NOTE Observer should live longer than comparator
  std::vector<Diff<T>> compare(const Shape &shape, const Reader<T> &expected,
                               const Reader<T> &obtained, Observer *observer = nullptr) const;

private:
  std::function<bool(T lhs, T rhs)> _compare_fn;
};

} // namespace tensor
} // namespace misc
} // namespace nnfw

#endif // __NNFW_MISC_TENSOR_COMPARATOR_H__
