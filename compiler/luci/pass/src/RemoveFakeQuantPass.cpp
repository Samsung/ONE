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

#include "luci/Pass/RemoveFakeQuantPass.h"

#include <luci/IR/CircleNodes.h>

namespace
{

bool remove_fake_quant(luci::CircleFakeQuant *fakequant)
{
  assert(fakequant != nullptr);

  auto input_node = loco::must_cast<luci::CircleNode *>(fakequant->inputs());

  replace(fakequant).with(input_node);

  return true;
}

} // namespace

namespace luci
{
/**
 * BEFORE
 *
 *    [CircleNode]
 *          |
 *    [CircleFakeQuant]
 *          |
 *    [CircleNode]
 *
 * AFTER
 *
 *    [CircleNode]
 *          |
 *    [CircleNode]   [CircleFakeQuant]
 *
 * CircleFakeQuant OP will be removed from the output graph
 */
bool RemoveFakeQuantPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto target_node = dynamic_cast<luci::CircleFakeQuant *>(node);
    if (target_node != nullptr)
      if (remove_fake_quant(target_node))
        changed = true;
  }
  return changed;
}

} // namespace luci
