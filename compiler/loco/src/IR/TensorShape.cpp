/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "loco/IR/TensorShape.h"

#include <cassert>

namespace loco
{

uint32_t element_count(const loco::TensorShape *tensor_shape)
{
  uint32_t res = 1;

  for (uint32_t axis = 0; axis < tensor_shape->rank(); ++axis)
  {
    // Let's use "assert" here as "caller" is responsible for this check.
    // Please refer to the header for details.
    assert(tensor_shape->dim(axis).known());
    res *= tensor_shape->dim(axis).value();
  }

  return res;
}

} // namespace loco

namespace loco
{

bool operator==(const TensorShape &lhs, const TensorShape &rhs)
{
  if (lhs.rank() != rhs.rank())
    return false;
  for (uint32_t axis = 0; axis < lhs.rank(); ++axis)
  {
    if (!(lhs.dim(axis) == rhs.dim(axis)))
      return false;
  }
  return true;
}

} // namespace loco
