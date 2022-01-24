/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

void connect(luci::ConnectNode *cn, const luci::CircleSVDF *node)
{
  auto *cloned = loco::must_cast<luci::CircleSVDF *>(cn->find_clone(node));

  luci::CircleNode *input = loco::must_cast<luci::CircleNode *>(node->input());
  luci::CircleNode *weight_feature = loco::must_cast<luci::CircleNode *>(node->weight_feature());
  luci::CircleNode *weight_time = loco::must_cast<luci::CircleNode *>(node->weight_time());
  luci::CircleNode *bias = loco::must_cast<luci::CircleNode *>(node->bias());
  luci::CircleNode *input_activation_state =
    loco::must_cast<luci::CircleNode *>(node->input_activation_state());

  cloned->input(cn->find_clone(input));
  cloned->weight_feature(cn->find_clone(weight_feature));
  cloned->weight_time(cn->find_clone(weight_time));
  cloned->bias(cn->find_clone(bias));
  cloned->input_activation_state(cn->find_clone(input_activation_state));
}

} // namespace

namespace luci
{

void ConnectNode::visit(const luci::CircleSVDF *node) { connect(this, node); }

} // namespace luci
