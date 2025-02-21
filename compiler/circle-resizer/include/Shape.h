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

#include <stdexcept>
#include <stdint.h>
#include <vector>

namespace circle_resizer
{
class Dim
{
public:
  explicit Dim(int32_t dim);

public:
  bool is_dynamic();
  int32_t value() const;
  bool operator==(const Dim &rhs) const;

private:
  // Note that in the future, we might need to support dimension with lower and upper bounds
  int32_t _dim_value;
};

using Shape = std::vector<Dim>;
} // namespace circle_resizer

#endif // __CIRCLE_RESIZER_SHAPE_H__
