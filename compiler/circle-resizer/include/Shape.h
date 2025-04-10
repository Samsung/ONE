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
  Shape(const std::initializer_list<Dim> &dims);
  Shape(const std::vector<Dim> &shape_vec);
  static Shape scalar();

public:
  size_t rank() const;
  Dim operator[](const size_t &axis) const;
  bool is_scalar() const;
  bool is_dynamic() const;
  bool operator==(const Shape &rhs) const;
  friend std::ostream &operator<<(std::ostream &os, const Shape &shape);

private:
  std::vector<Dim> _dims;
};

} // namespace circle_resizer

#endif // __CIRCLE_RESIZER_SHAPE_H__
