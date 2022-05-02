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

#include "luci/Pass/RemoveRedundantDequantizePass.h"

#include <luci/IR/CircleNodes.h>

namespace
{

bool remove_redundant_dequant(luci::CircleDequantize *dequant)
{
  assert(dequant != nullptr);

  auto prev = loco::must_cast<luci::CircleNode *>(dequant->input());
  if (prev->dtype() != loco::DataType::FLOAT32)
    return false;

  replace(dequant).with(prev);

  return true;
}

} // namespace

namespace luci
{
/**
 * Dequantize Op does the below things on the ifm.
 * 1. Element-wise update of quantized values (u8/s16) to fp32 values
 * 2. Update dtype to fp32
 * If the previous node is not quantized, dequantize Op is redundant.
 *
 * BEFORE
 *
 *     [CircleNode (A)]
 *            |
 *     [CircleNode (B)] (fp32)
 *            |
 *    [CircleDequantize]
 *            |
 *       [CircleNode]
 *
 * AFTER
 *
 *     [CircleNode (A)]
 *            |
 *     [CircleNode (B)] (fp32)
 *            |
 *       [CircleNode]
 */
bool RemoveRedundantDequantizePass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto target_node = dynamic_cast<luci::CircleDequantize *>(node);
    if (target_node != nullptr)
    {
      if (remove_redundant_dequant(target_node))
        changed = true;
    }
  }
  return changed;
}

} // namespace luci
