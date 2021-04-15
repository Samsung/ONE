/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ConnectNode.h"

namespace
{

void connect(luci::ConnectNode *cn, const luci::CirclePack *node)
{
  auto *cloned = loco::must_cast<luci::CirclePack *>(cn->find_clone(node));

  uint32_t values_count = cloned->values_count();
  for (uint32_t i = 0; i < values_count; ++i)
  {
    luci::CircleNode *value = loco::must_cast<luci::CircleNode *>(node->values(i));

    cloned->values(i, cn->find_clone(value));
  }
}

} // namespace

namespace luci
{

void ConnectNode::visit(const luci::CirclePack *node) { connect(this, node); }

} // namespace luci
