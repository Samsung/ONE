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
 * @file vector.h
 * @ingroup COM_AI_RUNTIME
 * @brief This file contains @c == operator to check equality of elements in two vectors
 */
#ifndef __NNFW_MISC_VECTOR_H__
#define __NNFW_MISC_VECTOR_H__

#include <vector>

/**
 * @brief       Compare elements of two vectors
 * @tparam T    Type of elements in vectors
 * @param[in] lhs   First vector to compare
 * @param[in] rhs   Second vector to compare
 * @return    @c true if all elements are equal, otherwise @c false.
 */
template <typename T> bool operator==(const std::vector<T> &lhs, const std::vector<T> &rhs)
{
  if (lhs.size() != rhs.size())
  {
    return false;
  }

  for (size_t ind = 0; ind < lhs.size(); ++ind)
  {
    if (lhs.at(ind) != rhs.at(ind))
    {
      return false;
    }
  }

  return true;
}

#endif // __NNFW_MISC_VECTOR_H__
