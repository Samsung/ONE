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

#include "luci/Pass/ReplaceSubWithAddPass.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Profile/CircleNodeOrigin.h>
#include <luci/Service/Nodes/CircleConst.h>

namespace
{

bool replace_sub_with_const_rhs(luci::CircleSub *sub)
{
  auto const_rhs = dynamic_cast<luci::CircleConst *>(sub->y());
  if (const_rhs == nullptr)
    return false;

  auto graph = sub->graph();

  auto neg_const_rhs = luci::clone(const_rhs);
  if (neg_const_rhs->dtype() == loco::DataType::FLOAT32)
  {
    for (uint32_t i = 0; i < neg_const_rhs->size<loco::DataType::FLOAT32>(); ++i)
      neg_const_rhs->at<loco::DataType::FLOAT32>(i) *= -1.0;
  }
  else
  {
    // TODO Support more data type
    return false;
  }

  auto add = graph->nodes()->create<luci::CircleAdd>();
  add->x(sub->x());
  add->y(neg_const_rhs);
  add->name(sub->name());
  add->fusedActivationFunction(sub->fusedActivationFunction());
  luci::add_origin(add, luci::get_origin(sub));
  loco::replace(sub).with(add);
  return true;
}

} // namespace

namespace luci
{

bool ReplaceSubWithAddPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto sub = dynamic_cast<luci::CircleSub *>(node))
    {
      if (replace_sub_with_const_rhs(sub))
        changed = true;
    }
  }

  return changed;
}

} // namespace luci
