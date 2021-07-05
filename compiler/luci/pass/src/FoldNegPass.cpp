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

#include "luci/Pass/FoldNegPass.h"

#include <luci/IR/CircleNodes.h>

namespace
{

bool fold_neg(luci::CircleNeg *neg)
{
  // Check neg has const input
  auto const_x = dynamic_cast<luci::CircleConst *>(neg->x());
  if (not const_x)
    return false;

  // If quantparam exists, fold by scale
  if (auto qparam = const_x->quantparam())
  {
    for (uint32_t i = 0; i < qparam->scale.size(); ++i)
      qparam->scale.at(i) *= -1.0;
  }
  else
  {
    switch (const_x->dtype())
    {
      case loco::DataType::FLOAT32:
        for (uint32_t i = 0; i < const_x->size<loco::DataType::FLOAT32>(); ++i)
          const_x->at<loco::DataType::FLOAT32>(i) *= -1.0;
        break;
      default:
        return false;
    }
  }

  loco::replace(neg).with(const_x);
  return true;
}

} // namespace

namespace luci
{

/**
 * Constant Folding for Neg Op
 **/
bool FoldNegPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto neg = dynamic_cast<luci::CircleNeg *>(node))
    {
      if (fold_neg(neg))
        changed = true;
    }
  }

  return changed;
}

} // namespace luci
