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

#include "luci/Pass/RemoveRedundantQuantizePass.h"

#include <luci/IR/CircleNode.h>

/**
 *  Remove redundant quantize operations. For subsequent Quantize Ops,
 *  only the last Quantize Op is valid, so we can remove the rest of the Quantize Op.
 *
 *  BEFORE
 *                                          [CircleNode_1]
 *                                                |
 *                             [CircleQuantize, dtype_1, scale_1, zero_point_1]
 *                                                |
 *                             [CircleQuantize, dtype_2, scale_2, zero_point_2]
 *                                                |
 *                                         [CircleNode_2]
 *
 *  AFTER
 *                                          [CircleNode_1]
 *                                         /              \
 *                                      /                    \
 *                                   /                          \
 *                                /                                \
 *                             /                                      \
 * [CircleQuantize, dtype_2, scale_2, zero_point_2] [CircleQuantize, dtype_1, scale_1, zero_point_1]
 *                          |
 *                   [CircleNode_2]
 *
 */

namespace
{

bool remove_redundant_quantize(luci::CircleQuantize *node)
{
  auto pred_node = loco::must_cast<luci::CircleNode *>(node->input());

  if (pred_node == nullptr)
    return false;

  if (node->quantparam() == nullptr or pred_node->quantparam() == nullptr)
    return false;

  if (node->quantparam()->scale.size() != 1 or node->quantparam()->zerop.size() != 1 or
      pred_node->quantparam()->scale.size() != 1 or pred_node->quantparam()->zerop.size() != 1)
  {
    return false;
  }

  if (node->dtype() != pred_node->dtype() or
      pred_node->quantparam()->scale.at(0) != node->quantparam()->scale.at(0) or
      pred_node->quantparam()->zerop.at(0) != node->quantparam()->zerop.at(0))
  {
    return false;
  }

  replace(node).with(pred_node);

  return true;
}

bool remove_redundant_subsequent_quantize(luci::CircleQuantize *node)
{
  auto pred_node = dynamic_cast<luci::CircleQuantize *>(node->input());
  if (pred_node == nullptr)
    return remove_redundant_quantize(node);

  node->input(pred_node->input());
  return true;
}

} // namespace

namespace luci
{

bool RemoveRedundantQuantizePass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto quantize_node = dynamic_cast<luci::CircleQuantize *>(node))
    {
      if (remove_redundant_subsequent_quantize(quantize_node))
        changed = true;
    }
  }
  return changed;
}

} // namespace luci
