/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/XpSepActFromTransposeConvPass.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeMixins.h>
#include <luci/Profile/CircleNodeOrigin.h>

namespace luci
{

/**
 * XpSepActFromTransposeConvPass
 * - Experimental Separate Activation From TransposeConv
 * - This pass exist temporary to separate activation function from
 * - TransposeConv to support backends that don't support this.
 * - This pass will be removed when all backends support fused activation.
 *
 *  BEFORE
 *       [Node]
 *         |
 *   [TransposeConv] (w/ Act)
 *         |
 *       [Node]
 *
 *  AFTER
 *
 *       [Node]
 *         |
 *   [TransposeConv]
 *         |
 *   [ReLU/ReLU6/...]
 *         |
 *       [Node]
 *
 */

namespace
{

bool separate_activation_fuction(luci::CircleTransposeConv *trconv)
{
  // cannot separate for quantized state: support F32 for now
  // TODO revise this to better way
  if (trconv->dtype() != loco::DataType::FLOAT32)
    return false;

  auto fused_act = trconv->fusedActivationFunction();
  if (fused_act == luci::FusedActFunc::NONE)
    return false;
  if (fused_act == luci::FusedActFunc::UNDEFINED)
    throw std::runtime_error("XpSepActFromTransposeConvPass Activation is undefined");

  // NOTE features() is call after replace().with();
  //      calling loco::replace(trconv).with(actnode) will also update actnode
  //      itself which will make totally wrong result with actnode input being
  //      itself. this happends as TransposeConv is re-used, not replaced with
  //      a new one.

  auto name = trconv->name();
  luci::CircleNode *actnode = nullptr;
  switch (fused_act)
  {
    case luci::FusedActFunc::RELU:
    {
      auto af = trconv->graph()->nodes()->create<luci::CircleRelu>();
      loco::replace(trconv).with(af);
      af->features(trconv);
      af->name(name + "/Relu");
      actnode = af;
    }
    break;
    case luci::FusedActFunc::RELU6:
    {
      auto af = trconv->graph()->nodes()->create<luci::CircleRelu6>();
      loco::replace(trconv).with(af);
      af->features(trconv);
      af->name(name + "/Relu6");
      actnode = af;
    }
    break;
    // TODO support more
    default:
      return false;
  }
  assert(actnode != nullptr);
  actnode->dtype(trconv->dtype());
  luci::add_origin(actnode, luci::get_origin(trconv));

  trconv->fusedActivationFunction(luci::FusedActFunc::NONE);

  return true;
}

} // namespace

bool XpSepActFromTransposeConvPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto trconv = dynamic_cast<luci::CircleTransposeConv *>(node);
    if (trconv != nullptr)
    {
      if (separate_activation_fuction(trconv))
        changed = true;
    }
  }

  return changed;
}

} // namespace luci
