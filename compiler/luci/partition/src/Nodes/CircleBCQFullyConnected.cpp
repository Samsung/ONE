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

void connect(luci::ConnectNode *cn, const luci::CircleBCQFullyConnected *node)
{
  auto *cloned = loco::must_cast<luci::CircleBCQFullyConnected *>(cn->find_clone(node));

  luci::CircleNode *input = loco::must_cast<luci::CircleNode *>(node->input());
  luci::CircleNode *weights_scales = loco::must_cast<luci::CircleNode *>(node->weights_scales());
  luci::CircleNode *weights_binary = loco::must_cast<luci::CircleNode *>(node->weights_binary());
  luci::CircleNode *bias = loco::must_cast<luci::CircleNode *>(node->bias());
  luci::CircleNode *weights_clusters =
    loco::must_cast<luci::CircleNode *>(node->weights_clusters());

  cloned->input(cn->find_clone(input));
  cloned->weights_scales(cn->find_clone(weights_scales));
  cloned->weights_binary(cn->find_clone(weights_binary));
  cloned->bias(cn->find_clone(bias));
  cloned->weights_clusters(cn->find_clone(weights_clusters));
}

} // namespace

namespace luci
{

void ConnectNode::visit(const luci::CircleBCQFullyConnected *node) { connect(this, node); }

} // namespace luci
