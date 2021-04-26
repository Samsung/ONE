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

void connect(luci::ConnectNode *cn, const luci::CircleMaxPool2D *node)
{
  auto *cloned = loco::must_cast<luci::CircleMaxPool2D *>(cn->find_clone(node));

  luci::CircleNode *value = loco::must_cast<luci::CircleNode *>(node->value());

  cloned->value(cn->find_clone(value));
}

} // namespace

namespace luci
{

void ConnectNode::visit(const luci::CircleMaxPool2D *node) { connect(this, node); }

} // namespace luci
