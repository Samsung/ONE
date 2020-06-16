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

#include "luci/Pass/ResolveCustomOpAddPass.h"

#include "flatbuffers/flexbuffers.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/AttrFusedActFunc.h>

namespace
{

/// @brief Returns the index of BroadcastTo node among cop's inputs.
// NOTE This function assumes there is only one BroadcastTo node among its inputs.
int32_t get_broadcastTo_index_among_inputs_of(luci::CircleCustom *cop)
{
  for (uint32_t idx = 0; idx < cop->numInputs(); idx++)
  {
    auto input = dynamic_cast<const luci::CircleCustomOut *>(cop->inputs(idx));
    if (input)
    {
      auto broadcastTo = dynamic_cast<luci::CircleCustom *>(input->input());
      assert(broadcastTo);
      if (broadcastTo->custom_code() == "BroadcastTo")
        return idx;
    }
  }

  return -1;
}

/** BEFORE
 *                                  [CircleConst]
 *                                        |
 *        [CircleNode]         [BroadcastTo(CircleCustom)]
 *              \                         |
 *               \                [CircleCustomOUt]
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

  auto input = dynamic_cast<const luci::CircleCustomOut *>(addv2->inputs(broadcastTo_idx));
  auto broadcastTo = dynamic_cast<luci::CircleCustom *>(input->input());

  auto add = addv2->graph()->nodes()->create<luci::CircleAdd>();
  add->fusedActivationFunction(luci::FusedActFunc::NONE);
  add->x(addv2->inputs(1 - broadcastTo_idx));
  add->y(broadcastTo->inputs(0));
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

  if (resolve_with_BroadcastTo(addv2))
    return true;

  auto add = addv2->graph()->nodes()->create<luci::CircleAdd>();
  add->fusedActivationFunction(luci::FusedActFunc::NONE);
  add->x(addv2->inputs(0));
  add->y(addv2->inputs(1));
  auto customOut = loco::succs(addv2);
  assert(customOut.size() == 1);
  replace(*customOut.begin()).with(add);

  return true;
}

} // namespace

namespace luci
{

bool ResolveCustomOpAddPass::run(loco::Graph *g)
{
  bool changed = false;

  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto cop = dynamic_cast<luci::CircleCustom *>(node);
    if (not cop)
      continue;

    changed |= resolve_custom_op(cop);
  }

  return changed;
}

} // namespace luci
