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

#include "CircleCloneNode.h"

#include <oops/UserExn.h>

// TODO relocate these to each files
namespace luci
{

luci::CircleNode *CloneNode::visit(const luci::CircleAdd *node)
{
  auto *cloned = _graph->nodes()->create<luci::CircleAdd>();
  cloned->fusedActivationFunction(node->fusedActivationFunction());
  return cloned;
}

luci::CircleNode *CloneNode::visit(const luci::CircleDiv *node)
{
  auto *cloned = _graph->nodes()->create<luci::CircleDiv>();
  cloned->fusedActivationFunction(node->fusedActivationFunction());
  return cloned;
}

luci::CircleNode *CloneNode::visit(const luci::CircleMean *node)
{
  auto *cloned = _graph->nodes()->create<luci::CircleMean>();
  cloned->keep_dims(node->keep_dims());
  return cloned;
}

luci::CircleNode *CloneNode::visit(const luci::CircleMul *node)
{
  auto *cloned = _graph->nodes()->create<luci::CircleMul>();
  cloned->fusedActivationFunction(node->fusedActivationFunction());
  return cloned;
}

luci::CircleNode *CloneNode::visit(const luci::CirclePow *)
{
  return _graph->nodes()->create<luci::CirclePow>();
}

luci::CircleNode *CloneNode::visit(const luci::CircleRsqrt *)
{
  return _graph->nodes()->create<luci::CircleRsqrt>();
}

luci::CircleNode *CloneNode::visit(const luci::CircleSqrt *)
{
  return _graph->nodes()->create<luci::CircleSqrt>();
}

luci::CircleNode *CloneNode::visit(const luci::CircleSquaredDifference *)
{
  return _graph->nodes()->create<luci::CircleSquaredDifference>();
}

luci::CircleNode *CloneNode::visit(const luci::CircleSub *node)
{
  auto *cloned = _graph->nodes()->create<luci::CircleSub>();
  cloned->fusedActivationFunction(node->fusedActivationFunction());
  return cloned;
}

} // namespace luci
