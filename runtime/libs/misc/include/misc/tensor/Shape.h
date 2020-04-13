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
 * @file Shape.h
 * @ingroup COM_AI_RUNTIME
 * @brief This file contains nnfw::misc::tensor::Shape class
 */

#ifndef __NNFW_MISC_TENSOR_SHAPE_H__
#define __NNFW_MISC_TENSOR_SHAPE_H__

#include <cstdint>
#include <cstddef>
#include <deque>
#include <initializer_list>
#include <ostream>
#include <string>
#include <cassert>

namespace nnfw
{
namespace misc
{
namespace tensor
{

/**
 * @brief Class to represent shape of a tensor
 */
class Shape
{
public:
  /**
   * @brief Construct a new Shape object
   * @param[in] rank    Rank of a tensor
   */
  Shape(uint32_t rank) { _dimensions.resize(rank); }

public:
  /**
   * @brief Construct a new Shape object
   * @param[in] dimensions    @c initializer_list<int32_t> of dimensions of tensor
   */
  Shape(const std::initializer_list<int32_t> &dimensions) : _dimensions{dimensions}
  {
    // Check overflow because initializer_list type can be larger size than max of uint32_t
    assert(dimensions.size() <= 0xFFFFFFFF);
  }

  /**
   * @brief Construct a new Shape object
   * @param[in] origin    @c Shape object to copy
   */
  Shape(const Shape &origin) = default;

public:
  /**
   * @brief Add dimension to the beginning
   * @param[in] d     dimension to add to the beginning
   * @return N/A
   */
  void prepend(int32_t d) { _dimensions.emplace_front(d); }

  /**
   * @brief Add dimension to the back
   * @param[in] d     dimension to add to the back
   * @return N/A
   */
  void append(int32_t d) { _dimensions.emplace_back(d); }

public:
  /**
   * @brief Get the rank of this shape
   * @return rank
   * @note   We can use static_cast\n
   *         because we don't support larger than max of uint32_t on constructor
   */
  uint32_t rank(void) const { return static_cast<uint32_t>(_dimensions.size()); }

public:
  /**
   * @brief Get specific dimension
   * @param[in] n  Index of dimension
   * @return n'th dimension
   */
  int32_t dim(uint32_t n) const { return _dimensions.at(n); }

  /**
   * @brief Get the reference of specific dimension
   * @param[in] n  Index of dimension
   * @return Reference of n'th dimension
   */
  int32_t &dim(uint32_t n) { return _dimensions.at(n); }

  const std::deque<int32_t> &dims() const { return _dimensions; }

public:
  /**
   * @brief Get the number of elements specified by this shape
   * @return The number of elements
   */
  uint64_t num_elements() const;

private:
  std::deque<int32_t> _dimensions;

public:
  /**
   * @brief Get a @c Shape object after parsing string
   * @param[in] s  String of dimension list. Accepted format is numbers separated by comma.
   * @return @c Shape object
   */
  static Shape from(const std::string &s);
};

/**
 * @brief Check equality of two @c Shape
 * @param[in] Shape First shape to compare
 * @param[in] Shape Second shape to compare
 * @return @c true if both shapes are equal, otherwise @c false
 */
bool operator==(const Shape &, const Shape &);

/**
 * @brief Send @c Shape to @c std::ostream
 * @param[in] os        @c std::ostream to process this @c Shape
 * @param[in] shape     @c Shape to send to @c ostream
 * @return Reference of @c std::ostream
 */
std::ostream &operator<<(std::ostream &os, const Shape &shape);

} // namespace tensor
} // namespace misc
} // namespace nnfw

#endif // __NNFW_MISC_TENSOR_SHAPE_H__
