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
 * @file Index.h
 * @ingroup COM_AI_RUNTIME
 * @brief This file contains nnfw::misc::tensor::Index struct
 */
#ifndef __NNFW_MISC_TENSOR_INDEX_H__
#define __NNFW_MISC_TENSOR_INDEX_H__

#include <cstdint>
#include <cstddef>

#include <vector>
#include <initializer_list>

namespace nnfw
{
namespace misc
{
namespace tensor
{

/**
 * @brief Struct to represent index of each dimension of a tensor
 */
struct Index
{
public:
  /**
   * @brief Construct a new @c Index object
   * @param[in] rank    Rank of a tensor
   */
  Index(uint32_t rank) { _offsets.resize(rank); }

public:
  /**
   * @brief Construct a new @c Index object
   * @param[in] offsets    Rank of a tensor of @c std::initializer_list<int32_t> type
   */
  Index(std::initializer_list<int32_t> offsets) : _offsets{offsets}
  {
    // DO NOTHING
  }

public:
  /**
   * @brief Get the rank
   * @return Rank that this @c Index object can handle
   * @note   We can use static_cast\n
   *         because size of _offsets is decieded by constructor's uintt_32 type argument
   */
  uint32_t rank(void) const { return static_cast<uint32_t>(_offsets.size()); }

public:
  /**
   * @brief Get the index n'th dimension
   * @param[in] n   Dimension
   * @return index of n'th dimension
   */
  int32_t at(uint32_t n) const { return _offsets.at(n); }

  /**
   * @brief Get the reference of the index n'th dimension
   * @param[in] n   Dimension
   * @return reference of index of n'th dimension
   */
  int32_t &at(uint32_t n) { return _offsets.at(n); }

private:
  std::vector<int32_t> _offsets;
};

/**
 * @brief Copy an @c Index with reversed order
 * @param[in] origin    @c Index object to copy
 * @return  an @c Index object with reversed order
 * @note    This is used to convert NNAPI tensor index to ARM tensor index or vice versa
 */
inline static Index copy_reverse(const Index &origin)
{
  uint32_t rank = origin.rank();
  Index target(rank);
  for (uint32_t i = 0; i < rank; i++)
    target.at(i) = origin.at(rank - 1 - i);
  return target;
}

} // namespace tensor
} // namespace misc
} // namespace nnfw

#endif // __NNFW_MISC_TENSOR_INDEX_H__
