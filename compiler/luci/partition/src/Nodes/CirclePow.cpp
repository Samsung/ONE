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

void connect(luci::ConnectNode *cn, const luci::CirclePow *node)
{
  auto *cloned = loco::must_cast<luci::CirclePow *>(cn->find_clone(node));

  luci::CircleNode *x = loco::must_cast<luci::CircleNode *>(node->x());
  luci::CircleNode *y = loco::must_cast<luci::CircleNode *>(node->y());

  cloned->x(cn->find_clone(x));
  cloned->y(cn->find_clone(y));
}

} // namespace

namespace luci
{

void ConnectNode::visit(const luci::CirclePow *node) { connect(this, node); }

} // namespace luci
