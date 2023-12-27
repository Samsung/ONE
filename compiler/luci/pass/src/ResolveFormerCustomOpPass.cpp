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

#include "luci/Pass/ResolveFormerCustomOpPass.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/AttrFusedActFunc.h>
#include <luci/Profile/CircleNodeOrigin.h>

#include <flatbuffers/flexbuffers.h>

namespace
{

bool resolve_with_BroadcastTo(luci::CircleCustom *node)
{
  // check if the number of inputs is 2.
  if (node->numInputs() != 2)
    return false;

  auto input = loco::must_cast<luci::CircleNode *>(node->inputs(0));

  // check if shape are support data types
  auto shape = loco::must_cast<luci::CircleNode *>(node->inputs(1));
  if (shape->dtype() != loco::DataType::S32 && shape->dtype() != loco::DataType::S64)
    return false;

  auto customOut = loco::succs(node);
  assert(customOut.size() == 1);

  // check if the data type of output is same with the one of the input feature map.
  auto output = loco::must_cast<luci::CircleNode *>(*customOut.begin());
  if (input->dtype() != output->dtype())
    return false;

  auto name = node->name();
  assert(name.length() > 0);

  auto broadcastTo = node->graph()->nodes()->create<luci::CircleBroadcastTo>();
  broadcastTo->input(input);
  broadcastTo->shape(shape);
  broadcastTo->name(name);
  luci::add_origin(broadcastTo, luci::get_origin(node));

  replace(*customOut.begin()).with(broadcastTo);

  return true;
}

bool resolve_custom_op(luci::CircleCustom *node)
{
  const std::string custom_code = node->custom_code();

  if (custom_code == "BroadcastTo")
  {
    return resolve_with_BroadcastTo(node);
  }
  // TODO add more custom codes

  return false;
}

} // namespace

namespace luci
{

bool ResolveFormerCustomOpPass::run(loco::Graph *g)
{
  bool changed = false;

  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto cop = dynamic_cast<luci::CircleCustom *>(node);
    if (not cop)
      continue;

    if (resolve_custom_op(cop))
      changed = true;
  }

  return changed;
}

} // namespace luci
