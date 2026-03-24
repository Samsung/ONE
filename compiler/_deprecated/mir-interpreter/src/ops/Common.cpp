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

#include <cassert>

#include "Common.h"

namespace mir_interpreter
{

using namespace mir;

Index shift(const Index &in_index, const Shape &shift_from)
{
  Index index = in_index;
  assert(index.rank() == shift_from.rank());
  for (int32_t d = 0; d < in_index.rank(); ++d)
  {
    index.at(d) = index.at(d) + shift_from.dim(d);
  }
  return index;
}

} // namespace mir_interpreter
