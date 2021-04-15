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

void connect(luci::ConnectNode *cn, const luci::CircleLocalResponseNormalization *node)
{
  auto *cloned = loco::must_cast<luci::CircleLocalResponseNormalization *>(cn->find_clone(node));

  luci::CircleNode *input = loco::must_cast<luci::CircleNode *>(node->input());

  cloned->input(cn->find_clone(input));
}

} // namespace

namespace luci
{

void ConnectNode::visit(const luci::CircleLocalResponseNormalization *node) { connect(this, node); }

} // namespace luci
