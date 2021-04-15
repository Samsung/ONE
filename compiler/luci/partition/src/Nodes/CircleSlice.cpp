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

void connect(luci::ConnectNode *cn, const luci::CircleSlice *node)
{
  auto *cloned = loco::must_cast<luci::CircleSlice *>(cn->find_clone(node));

  luci::CircleNode *input = loco::must_cast<luci::CircleNode *>(node->input());
  luci::CircleNode *begin = loco::must_cast<luci::CircleNode *>(node->begin());
  luci::CircleNode *size = loco::must_cast<luci::CircleNode *>(node->size());

  cloned->input(cn->find_clone(input));
  cloned->begin(cn->find_clone(begin));
  cloned->size(cn->find_clone(size));
}

} // namespace

namespace luci
{

void ConnectNode::visit(const luci::CircleSlice *node) { connect(this, node); }

} // namespace luci
