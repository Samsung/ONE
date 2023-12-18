/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

luci::CircleNode *CloneNode::visit(const luci::CircleGRU *node)
{
  if (node->fusedActivationFunction() == luci::FusedActFunc::UNDEFINED)
    return nullptr;

  auto *cloned = _graph->nodes()->create<luci::CircleGRU>();
  if (cloned != nullptr)
  {
    cloned->fusedActivationFunction(node->fusedActivationFunction());
    cloned->returnSequences(node->returnSequences());
    cloned->timeMajor(node->timeMajor());
  }
  return cloned;
}

} // namespace luci
