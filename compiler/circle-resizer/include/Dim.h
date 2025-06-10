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

#ifndef __CIRCLE_RESIZER_DIM_H__
#define __CIRCLE_RESIZER_DIM_H__

#include <stdint.h>

namespace circle_resizer
{

/**
 * The representation of a single dimension. Note that a dimension can be dynamic.
 */
class Dim
{
public:
  /**
   * @brief Initialize single dimension. Note that '-1' means a dynamic dimension.
   *
   * Exceptions:
   * - std::runtime_error if provided dim value is less than -1.
   */
  explicit Dim(int32_t dim);

  /**
   * @brief Create dynamic dimension. Note that it's equivalent of Dim{-1}.
   */
  static Dim dynamic();

public:
  /**
   * @brief Returns true if the dimension is dynamic. Otherwise, return false.
   */
  bool is_dynamic() const;

  /**
   * @brief Returns value of dimension in int32_t representation.
   */
  int32_t value() const;

  /**
   * @brief Returns true of the current dimension and the provided rhs are equal.
   */
  bool operator==(const Dim &rhs) const;

private:
  // Note that in the future, we might need to support dimension with lower and upper bounds
  int32_t _dim_value;
};

} // namespace circle_resizer

#endif // __CIRCLE_RESIZER_DIM_H__
