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

#include "luci/IR/Nodes/CircleInput.h"

#include <cassert>
#include <limits>

namespace luci
{

void CircleInput::index(const loco::GraphInputIndex &index)
{
  // CircleInput internally stores "GraphInputIndex" as int64_t
  _index = static_cast<int64_t>(index);
}

loco::GraphInputIndex CircleInput::index(void) const
{
  assert(_index >= std::numeric_limits<loco::GraphInputIndex>::min());
  assert(_index <= std::numeric_limits<loco::GraphInputIndex>::max());
  return static_cast<loco::GraphInputIndex>(_index);
}

} // namespace luci
