/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Support.Misc.h"

namespace fme_apply
{

void copy_shape(luci::CircleNode *from, luci::CircleNode *to)
{
  if (not from)
    throw std::invalid_argument("from");

  if (not to)
    throw std::invalid_argument("to");

  to->rank(from->rank());
  for (uint32_t i = 0; i < from->rank(); ++i)
  {
    to->dim(i) = from->dim(i);
  }
}

} // namespace fme_apply
