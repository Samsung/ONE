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

void connect(luci::ConnectNode *cn, const luci::CircleGelu *node)
{
  auto *cloned = loco::must_cast<luci::CircleGelu *>(cn->find_clone(node));

  luci::CircleNode *features = loco::must_cast<luci::CircleNode *>(node->features());

  cloned->features(cn->find_clone(features));
}

} // namespace

namespace luci
{

void ConnectNode::visit(const luci::CircleGelu *node) { connect(this, node); }

} // namespace luci
