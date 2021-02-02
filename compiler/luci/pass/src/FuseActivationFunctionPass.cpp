/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/FuseActivationFunctionPass.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleOpcode.h>

namespace luci
{

bool fuse_activation_function(luci::CircleNode *node)
{
  auto preds = loco::preds(node);
  assert(preds.size() == 1);

  auto pred_node = static_cast<luci::CircleNode *>(*preds.begin());
  if (loco::succs(pred_node).size() != 1)
    return false;

  auto node_with_fused_act =
    dynamic_cast<luci::LuciNodeMixin<luci::LuciNodeTrait::FusedActFunc> *>(pred_node);
  if (node_with_fused_act == nullptr)
    return false;

  // TODO remove this work-around
  // This will skip fuse for concat as luci-interpreter doesn't support this yet
  if (dynamic_cast<luci::CircleConcatenation *>(pred_node) != nullptr)
    return false;

  auto fused_act = node_with_fused_act->fusedActivationFunction();

  luci::FusedActFunc target_func = luci::FusedActFunc::UNDEFINED;

  auto opcode = node->opcode();
  if (opcode == luci::CircleOpcode::RELU)
  {
    if (fused_act == luci::FusedActFunc::NONE || fused_act == luci::FusedActFunc::RELU)
      target_func = luci::FusedActFunc::RELU;
    else if (fused_act == luci::FusedActFunc::RELU6)
      target_func = luci::FusedActFunc::RELU6;
    else
      return false;
  }
  else if (opcode == luci::CircleOpcode::RELU6)
  {
    if (fused_act == luci::FusedActFunc::NONE || fused_act == luci::FusedActFunc::RELU ||
        fused_act == luci::FusedActFunc::RELU6)
      target_func = luci::FusedActFunc::RELU6;
    else
      return false;
  }
  else if (opcode == luci::CircleOpcode::RELU_N1_TO_1)
  {
    if (fused_act == luci::FusedActFunc::NONE || fused_act == luci::FusedActFunc::RELU_N1_TO_1)
      target_func = luci::FusedActFunc::RELU_N1_TO_1;
    else
      return false;
  }
  else if (opcode == luci::CircleOpcode::TANH)
  {
    if (fused_act == luci::FusedActFunc::NONE)
      target_func = luci::FusedActFunc::TANH;
    else
      return false;
  }
  else
    return false;

  node_with_fused_act->fusedActivationFunction(target_func);
  loco::replace(node).with(pred_node);

  node->drop();

  return true;
}

bool FuseActivationFunctionPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto circle_node = static_cast<luci::CircleNode *>(node);
    auto opcode = circle_node->opcode();
    if (opcode == luci::CircleOpcode::RELU || opcode == luci::CircleOpcode::RELU6 ||
        opcode == luci::CircleOpcode::RELU_N1_TO_1 || opcode == luci::CircleOpcode::TANH)
    {
      if (fuse_activation_function(circle_node))
        changed = true;
    }
  }

  return changed;
}

} // namespace luci
