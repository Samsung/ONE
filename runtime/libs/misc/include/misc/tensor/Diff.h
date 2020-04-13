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
 * @file Diff.h
 * @ingroup COM_AI_RUNTIME
 * @brief This file contains nnfw::misc::tensor::Diff struct
 */

#ifndef __NNFW_MISC_TENSOR_DIFF_H__
#define __NNFW_MISC_TENSOR_DIFF_H__

#include "misc/tensor/Index.h"

namespace nnfw
{
namespace misc
{
namespace tensor
{

/**
 * @brief Struct to have information after comparing two elements of two tensors
 */
template <typename T> struct Diff
{
  Index index; /**< Index of elements in two tensors, which turn out to be different */

  T expected; /**< Expected value of element of first tensor */
  T obtained; /**< Obtained value of element of second tensor */

  /**
   * @brief Construct a new @c Diff object
   * @param[in] i   Initial value of index
   */
  Diff(const Index &i) : index(i)
  {
    // DO NOTHING
  }

  /**
   * @brief Construct a new @c Diff object
   * @param[in] i   Index value
   * @param[in] e   Expected value of element of first tensor
   * @param[in] o   Obtained value of element of second tensor
   */
  Diff(const Index &i, const T &e, const T &o) : index(i), expected{e}, obtained{o}
  {
    // DO NOTHING
  }
};

} // namespace tensor
} // namespace misc
} // namespace nnfw

#endif // __NNFW_MISC_TENSOR_DIFF_H__
