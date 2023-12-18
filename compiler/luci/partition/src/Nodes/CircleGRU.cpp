/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

void connect(luci::ConnectNode *cn, const luci::CircleGRU *node)
{
  auto *cloned = loco::must_cast<luci::CircleGRU *>(cn->find_clone(node));

  luci::CircleNode *input = loco::must_cast<luci::CircleNode *>(node->input());
  luci::CircleNode *hidden_input = loco::must_cast<luci::CircleNode *>(node->hidden_input());
  luci::CircleNode *hidden_hidden = loco::must_cast<luci::CircleNode *>(node->hidden_hidden());
  luci::CircleNode *state = loco::must_cast<luci::CircleNode *>(node->state());

  cloned->input(cn->find_clone(input));
  cloned->hidden_input(cn->find_clone(hidden_input));
  cloned->hidden_hidden(cn->find_clone(hidden_hidden));
  cloned->state(cn->find_clone(state));
}

} // namespace

namespace luci
{

void ConnectNode::visit(const luci::CircleGRU *node) { connect(this, node); }

} // namespace luci
