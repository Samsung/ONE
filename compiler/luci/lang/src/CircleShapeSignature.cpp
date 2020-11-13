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

#include "luci/IR/CircleShapeSignature.h"

namespace luci
{

bool operator==(const ShapeSignature &lhs, const ShapeSignature &rhs)
{
  if (lhs.rank() != rhs.rank())
    return false;

  for (uint32_t i = 0; i < lhs.rank(); ++i)
    if (lhs.dim(i) != rhs.dim(i))
      return false;

  return true;
}

} // namespace luci
