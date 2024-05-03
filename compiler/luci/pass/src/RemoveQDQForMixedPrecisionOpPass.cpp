/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/RemoveQDQForMixedPrecisionOpPass.h"

#include <luci/IR/CircleNode.h>

/**
 *  Remove Quantize-Dequantize pattern for backends with mixed-precision operator.
 *
 *  BEFORE
 *                                          [CircleNode_1]
 *                                                |
 *                             [CircleQuantize, dtype_1, scale_1, zero_point_1]
 *                                                |
 *                                       [CircleDequantize]
 *                                                |
 *                             [CircleQuantize, dtype_2, scale_2, zero_point_2]
 *                                                |
 *                                       [CircleDequantize]
 *                                                |
 *                                         [CircleNode_2]
 *
 *  AFTER
 *
 *                                          [CircleNode_1]
 *                                                |
 *                             [CircleQuantize, dtype_2, scale_2, zero_point_2]
 *                                                |
 *                                       [CircleDequantize]
 *                                                |
 *                                         [CircleNode_2]
 *
 */

namespace
{

bool remove_qdq_for_mpo(luci::CircleDequantize *node)
{
  auto prev = dynamic_cast<luci::CircleQuantize *>(node->input());
  if (not prev)
    return false;

  auto prev_prev = dynamic_cast<luci::CircleDequantize *>(prev->input());
  if (not prev_prev)
    return false;

  auto prev_prev_prev = dynamic_cast<luci::CircleQuantize *>(prev_prev->input());
  if (not prev_prev_prev)
    return false;

  auto input = loco::must_cast<luci::CircleNode *>(prev_prev_prev->input());

  const static std::set<luci::CircleOpcode> supported_ops{luci::CircleOpcode::FULLY_CONNECTED,
                                                          luci::CircleOpcode::BATCH_MATMUL};

  if (supported_ops.find(input->opcode()) == supported_ops.end())
    return false;

  prev->input(input);

  return true;
}

} // namespace

namespace luci
{

bool RemoveQDQForMixedPrecisionOpPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::postorder_traversal(loco::output_nodes(g)))
  {
    if (auto dq = dynamic_cast<luci::CircleDequantize *>(node))
    {
      if (remove_qdq_for_mpo(dq))
        changed = true;
    }
  }
  return changed;
}

} // namespace luci
