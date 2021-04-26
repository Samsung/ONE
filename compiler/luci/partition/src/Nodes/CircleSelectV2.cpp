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

void connect(luci::ConnectNode *cn, const luci::CircleSelectV2 *node)
{
  auto *cloned = loco::must_cast<luci::CircleSelectV2 *>(cn->find_clone(node));

  luci::CircleNode *condition = loco::must_cast<luci::CircleNode *>(node->condition());
  luci::CircleNode *t = loco::must_cast<luci::CircleNode *>(node->t());
  luci::CircleNode *e = loco::must_cast<luci::CircleNode *>(node->e());

  cloned->condition(cn->find_clone(condition));
  cloned->t(cn->find_clone(t));
  cloned->e(cn->find_clone(e));
}

} // namespace

namespace luci
{

void ConnectNode::visit(const luci::CircleSelectV2 *node) { connect(this, node); }

} // namespace luci
