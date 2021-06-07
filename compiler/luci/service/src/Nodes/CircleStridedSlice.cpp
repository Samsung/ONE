/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "CircleCloneNode.h"

namespace luci
{

luci::CircleNode *CloneNodeLet<CN::STUV>::visit(const luci::CircleStridedSlice *node)
{
  auto *cloned = _graph->nodes()->create<luci::CircleStridedSlice>();
  if (cloned != nullptr)
  {
    cloned->begin_mask(node->begin_mask());
    cloned->end_mask(node->end_mask());
    cloned->ellipsis_mask(node->ellipsis_mask());
    cloned->new_axis_mask(node->new_axis_mask());
    cloned->shrink_axis_mask(node->shrink_axis_mask());
  }
  return cloned;
}

} // namespace luci
