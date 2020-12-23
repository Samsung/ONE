/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LOCO_IR_DIMENSION_H__
#define __LOCO_IR_DIMENSION_H__

#include <cstdint>

namespace loco
{

/**
 * @brief The value of one dimension in a tensor shape
 * @note The value may be unknown
 */
class Dimension final
{
private:
  enum class Kind
  {
    Known,
    Unknown
  };

public:
  /// @brief Construct an "unknown" dimension
  Dimension() = default;

  /// @brief Construct a "known" dimension
  Dimension(uint32_t value) { set(value); }

public:
  /// @brief Return whether the value is known (or not)
  bool known(void) const { return _kind == Kind::Known; }

  /// @brief Return the value
  /// @note This value is meaningful only for known dimension
  uint32_t value(void) const { return _value; }

  void set(uint32_t value)
  {
    _kind = Kind::Known;
    _value = value;
  }

  void unset(void)
  {
    _kind = Kind::Unknown;
    _value = 0;
  }

private:
  Kind _kind{Kind::Unknown};
  uint32_t _value{0};
};

/**
 * @brief Equality operator between two Dimensions
 *
 * @note Refer to the definition of equality of dimemsion at
 *       https://www.tensorflow.org/api_docs/python/tf/Dimension#__eq__
 */
bool operator==(const Dimension &, const Dimension &);
bool operator==(const Dimension &, uint32_t);
bool operator==(uint32_t, const Dimension &);

/// @brief Make an "unknown" dimension
Dimension make_dimension(void);

} // namespace loco

#endif // __LOCO_IR_DIMENSION_H__
