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

void connect(luci::ConnectNode *cn, const luci::CircleBCQGather *node)
{
  auto *cloned = loco::must_cast<luci::CircleBCQGather *>(cn->find_clone(node));

  luci::CircleNode *input_scales = loco::must_cast<luci::CircleNode *>(node->input_scales());
  luci::CircleNode *input_binary = loco::must_cast<luci::CircleNode *>(node->input_binary());
  luci::CircleNode *indices = loco::must_cast<luci::CircleNode *>(node->indices());
  luci::CircleNode *input_clusters = loco::must_cast<luci::CircleNode *>(node->input_clusters());

  cloned->input_scales(cn->find_clone(input_scales));
  cloned->input_binary(cn->find_clone(input_binary));
  cloned->indices(cn->find_clone(indices));
  cloned->input_clusters(cn->find_clone(input_clusters));
}

} // namespace

namespace luci
{

void ConnectNode::visit(const luci::CircleBCQGather *node) { connect(this, node); }

} // namespace luci
