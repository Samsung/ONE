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

#include "luci/Pass/RemoveQuantDequantSeqPass.h"

#include <luci/IR/CircleNodes.h>

namespace
{

bool remove_quant_dequant(luci::CircleDequantize *dequant)
{
  assert(dequant != nullptr);

  auto quantize = dynamic_cast<luci::CircleQuantize *>(dequant->input());
  if (quantize == nullptr)
    return false;

  auto input_node = loco::must_cast<luci::CircleNode *>(quantize->input());

  replace(dequant).with(input_node);

  return true;
}

} // namespace

namespace luci
{
/**
 * BEFORE
 *
 *       [CircleNode]
 *            |
 *     [CircleQuantize]
 *            |
 *    [CircleDequantize]
 *            |
 *       [CircleNode]
 *
 * AFTER
 *
 *    [CircleNode]    [CircleQuantize]
 *          |                |
 *    [CircleNode]   [CircleDequantize]
 *
 * CircleQuant-CircleDequant sequance will be removed from the output graph
 */
bool RemoveQuantDequantSeqPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto target_node = dynamic_cast<luci::CircleDequantize *>(node);
    if (target_node != nullptr)
    {
      if (remove_quant_dequant(target_node))
        changed = true;
    }
  }
  return changed;
}

} // namespace luci
