/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ir/TypeInfo.h"

namespace onert
{
namespace ir
{

bool operator==(const TypeInfo &lhs, const TypeInfo &rhs)
{
  if (lhs.type() != rhs.type())
  {
    return false;
  }

  if (lhs.zero_point() != rhs.zero_point())
  {
    return false;
  }

  if (lhs.scale() != rhs.scale())
  {
    return false;
  }

  return true;
}

bool operator!=(const TypeInfo &lhs, const TypeInfo &rhs) { return !(lhs == rhs); }

} // namespace ir
} // namespace onert
