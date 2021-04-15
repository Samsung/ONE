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

void connect(luci::ConnectNode *cn, const luci::CircleOneHot *node)
{
  auto *cloned = loco::must_cast<luci::CircleOneHot *>(cn->find_clone(node));

  luci::CircleNode *indices = loco::must_cast<luci::CircleNode *>(node->indices());
  luci::CircleNode *depth = loco::must_cast<luci::CircleNode *>(node->depth());
  luci::CircleNode *on_value = loco::must_cast<luci::CircleNode *>(node->on_value());
  luci::CircleNode *off_value = loco::must_cast<luci::CircleNode *>(node->off_value());

  cloned->indices(cn->find_clone(indices));
  cloned->depth(cn->find_clone(depth));
  cloned->on_value(cn->find_clone(on_value));
  cloned->off_value(cn->find_clone(off_value));
}

} // namespace

namespace luci
{

void ConnectNode::visit(const luci::CircleOneHot *node) { connect(this, node); }

} // namespace luci
