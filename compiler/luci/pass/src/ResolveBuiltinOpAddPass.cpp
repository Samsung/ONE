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

#include "luci/Pass/ResolveBuiltinOpAddPass.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/AttrFusedActFunc.h>
#include <luci/Profile/CircleNodeOrigin.h>

#include <flatbuffers/flexbuffers.h>
namespace
{

/// @brief Returns the index of BroadcastTo node among cop's inputs.
// NOTE This function assumes there is only one BroadcastTo node among its inputs.
int32_t get_broadcastTo_index_among_inputs_of(luci::CircleCustom *cop)
{
  for (uint32_t idx = 0; idx < cop->numInputs(); idx++)
  {
    auto broadcastTo = dynamic_cast<luci::CircleBroadcastTo *>(cop->inputs(idx));
    if (broadcastTo)
      return idx;
  }

  return -1;
}

/** BEFORE
 *
 *
 *        [CircleNode]            [CircleConst]
 *              \                      |
 *               \           [CircleBroadcastTo(Builtin)]
 *                \                   /
 *               [AddV2(CircleCustom)]
 *  AFTER
 *
 *         [CircleConst]         [CircleNode]
 *                   \           /
 *                    \         /
 *                    [CircleAdd]
 */
bool resolve_with_BroadcastTo(luci::CircleCustom *addv2)
{
  int32_t broadcastTo_idx = get_broadcastTo_index_among_inputs_of(addv2);

  if (broadcastTo_idx == -1)
    return false;

  auto name = addv2->name();
  assert(name.length() > 0);

  auto add = addv2->graph()->nodes()->create<luci::CircleAdd>();

  auto brodcastTo =
    loco::must_cast<const luci::CircleBroadcastTo *>(addv2->inputs(broadcastTo_idx));

  add->fusedActivationFunction(luci::FusedActFunc::NONE);
  add->x(addv2->inputs(1 - broadcastTo_idx));
  add->y(brodcastTo->input());
  add->name(name + "/Add");
  luci::add_origin(add,
                   luci::composite_origin({luci::get_origin(brodcastTo), luci::get_origin(addv2)}));

  auto customOut = loco::succs(addv2);
  assert(customOut.size() == 1);
  replace(*customOut.begin()).with(add);

  return true;
}

bool resolve_custom_op(luci::CircleCustom *addv2)
{
  const std::string custom_code = addv2->custom_code();
  const std::vector<uint8_t> custom_options = addv2->custom_options();

  if (custom_code != "AddV2")
    return false;

  if (addv2->numInputs() != 2)
    return false;

  for (uint32_t i = 0; i < addv2->numInputs(); i++)
  {
    // check if inputs are nullptr
    if (addv2->inputs(i) == nullptr)
      return false;

    // check if inputs are suppport data types
    auto input = loco::must_cast<luci::CircleNode *>(addv2->inputs(i));
    switch (input->dtype())
    {
      case loco::DataType::U8:
      case loco::DataType::S8:
      case loco::DataType::S16:
      case loco::DataType::S32:
      case loco::DataType::FLOAT32:
        break;
      default:
        return false;
    }
  }
  if (resolve_with_BroadcastTo(addv2))
    return true;

  auto name = addv2->name();
  assert(name.length() > 0);

  auto add = addv2->graph()->nodes()->create<luci::CircleAdd>();
  add->fusedActivationFunction(luci::FusedActFunc::NONE);
  add->x(addv2->inputs(0));
  add->y(addv2->inputs(1));
  add->name(name + "/Add");
  luci::add_origin(add, luci::get_origin(addv2));

  auto customOut = loco::succs(addv2);
  assert(customOut.size() == 1);
  replace(*customOut.begin()).with(add);

  return true;
}

} // namespace

namespace luci
{

bool ResolveBuiltinOpAddPass::run(loco::Graph *g)
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
