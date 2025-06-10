/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __CIRCLE_RESIZER_SHAPE_H__
#define __CIRCLE_RESIZER_SHAPE_H__

#include "Dim.h"

#include <ostream>
#include <vector>

namespace circle_resizer
{

/**
 * The representation of a single shape.
 */
class Shape
{
public:
  /**
   * @brief Initialize shape with initializer list of dims.
   */
  Shape(const std::initializer_list<Dim> &dims);

  /**
   * @brief Initialize shape with vector of dims.
   */
  Shape(const std::vector<Dim> &shape_vec);

  /**
   * @brief Initialize static shape with initializer list of of uint32_t values.
   *
   * Exceptions:
   * - std::out_of_range if some elements in shape_vec exceed int32_t range.
   */
  Shape(const std::initializer_list<uint32_t> &shape_vec);

  /**
   * @brief Create scalar shape. Note, that the same can be achieved with Shape{}.
   */
  static Shape scalar();

public:
  /**
   * @brief Returns number of dimensions in the shape.
   */
  size_t rank() const;

  /**
   * @brief Returns dimension of the position determined by axis.
   *
   * Exceptions:
   * - std::invalid_argument if the method is called on a scalar shape.
   * - std::out_of_range if the provided axis is greater than rank.
   */
  Dim operator[](const size_t &axis) const;

  /**
   * @brief Returns true if the shape is a scalar. Otherwise, return false.
   */
  bool is_scalar() const;

  /**
   * @brief Returns true if all dimensions in the shape are static or the shape is a scalar.
   *        Otherwise, return false.
   */
  bool is_dynamic() const;

  /**
   * @brief Returns true of the current shape and the provided rhs are equal.
   */
  bool operator==(const Shape &rhs) const;

private:
  std::vector<Dim> _dims;
};

/**
 * @brief Print the shape in format [1, 2, 3].
 */
std::ostream &operator<<(std::ostream &os, const Shape &shape);

} // namespace circle_resizer

#endif // __CIRCLE_RESIZER_SHAPE_H__
