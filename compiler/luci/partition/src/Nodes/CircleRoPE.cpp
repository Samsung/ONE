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

#include "luci/ConnectNode.h"

namespace
{

void connect(luci::ConnectNode *cn, const luci::CircleRoPE *node)
{
  auto *cloned = loco::must_cast<luci::CircleRoPE *>(cn->find_clone(node));

  luci::CircleNode *input = loco::must_cast<luci::CircleNode *>(node->input());
  luci::CircleNode *sin_table = loco::must_cast<luci::CircleNode *>(node->sin_table());
  luci::CircleNode *cos_table = loco::must_cast<luci::CircleNode *>(node->cos_table());

  cloned->input(cn->find_clone(input));
  cloned->sin_table(cn->find_clone(sin_table));
  cloned->cos_table(cn->find_clone(cos_table));
}

} // namespace

namespace luci
{

void ConnectNode::visit(const luci::CircleRoPE *node) { connect(this, node); }

} // namespace luci
