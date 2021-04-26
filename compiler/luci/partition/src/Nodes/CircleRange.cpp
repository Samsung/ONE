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

void connect(luci::ConnectNode *cn, const luci::CircleRange *node)
{
  auto *cloned = loco::must_cast<luci::CircleRange *>(cn->find_clone(node));

  luci::CircleNode *start = loco::must_cast<luci::CircleNode *>(node->start());
  luci::CircleNode *limit = loco::must_cast<luci::CircleNode *>(node->limit());
  luci::CircleNode *delta = loco::must_cast<luci::CircleNode *>(node->delta());

  cloned->start(cn->find_clone(start));
  cloned->limit(cn->find_clone(limit));
  cloned->delta(cn->find_clone(delta));
}

} // namespace

namespace luci
{

void ConnectNode::visit(const luci::CircleRange *node) { connect(this, node); }

} // namespace luci
