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

namespace luci
{

void ConnectNode::visit(const luci::CircleBatchToSpaceND *node)
{
  auto *cloned = loco::must_cast<luci::CircleBatchToSpaceND *>(find_clone(node));
  luci::CircleNode *input = loco::must_cast<luci::CircleNode *>(node->input());
  luci::CircleNode *block_shape = loco::must_cast<luci::CircleNode *>(node->block_shape());
  luci::CircleNode *crops = loco::must_cast<luci::CircleNode *>(node->crops());
  cloned->input(find_clone(input));
  cloned->block_shape(find_clone(block_shape));
  cloned->crops(find_clone(crops));
}

} // namespace luci
