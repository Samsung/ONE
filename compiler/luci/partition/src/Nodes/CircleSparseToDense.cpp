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

void connect(luci::ConnectNode *cn, const luci::CircleSparseToDense *node)
{
  auto *cloned = loco::must_cast<luci::CircleSparseToDense *>(cn->find_clone(node));

  luci::CircleNode *indices = loco::must_cast<luci::CircleNode *>(node->indices());
  luci::CircleNode *output_shape = loco::must_cast<luci::CircleNode *>(node->output_shape());
  luci::CircleNode *values = loco::must_cast<luci::CircleNode *>(node->values());
  luci::CircleNode *default_value = loco::must_cast<luci::CircleNode *>(node->default_value());

  cloned->indices(cn->find_clone(indices));
  cloned->output_shape(cn->find_clone(output_shape));
  cloned->values(cn->find_clone(values));
  cloned->default_value(cn->find_clone(default_value));
}

} // namespace

namespace luci
{

void ConnectNode::visit(const luci::CircleSparseToDense *node) { connect(this, node); }

} // namespace luci
