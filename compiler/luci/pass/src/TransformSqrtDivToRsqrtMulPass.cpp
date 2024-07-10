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

#include "luci/Pass/TransformSqrtDivToRsqrtMulPass.h"

#include "helpers/NodeFiller.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Profile/CircleNodeOrigin.h>

namespace
{

/**
 *  BEFORE
 *        [CircleNode] [CircleNode]
 *              |           |
 *              |      [CircleSqrt]
 *              |       |
 *             [CircleDiv]
 *                  |
 *             [CircleNode]
 *
 *  AFTER
 *        [CircleNode] [CircleNode]
 *              |           |
 *              |      [CircleRsqrt]   [CircleSqrt]
 *              |       |                   |
 *             [CircleMul]             [CircleDiv]
 *                  |
 *             [CircleNode]
 *
 */

bool transform_sqrtdiv_to_rsqrtmul(luci::CircleDiv *div)
{
  assert(div != nullptr);

  // skip if x is const, for FuseRsqrtPass
  auto *const_node = dynamic_cast<luci::CircleConst *>(div->x());
  if (const_node != nullptr)
    return false;

  auto *sqrt = dynamic_cast<luci::CircleSqrt *>(div->y());
  if (sqrt == nullptr)
    return false;

  auto *graph = div->graph();

  auto *rsqrt = graph->nodes()->create<luci::CircleRsqrt>();
  rsqrt->x(sqrt->x());
  rsqrt->name(sqrt->name() + "_rsqrt");
  luci::add_origin(rsqrt, luci::get_origin(sqrt));

  auto *mul = graph->nodes()->create<luci::CircleMul>();
  mul->x(div->x());
  mul->y(rsqrt);
  mul->fusedActivationFunction(div->fusedActivationFunction());
  mul->name(div->name() + "_mul");
  luci::add_origin(mul, luci::get_origin(div));

  replace(div).with(mul);

  return true;
}

} // namespace

namespace luci
{

bool TransformSqrtDivToRsqrtMulPass::run(loco::Graph *g)
{
  bool changed = false;

  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto div = dynamic_cast<luci::CircleDiv *>(node))
    {
      if (transform_sqrtdiv_to_rsqrtmul(div))
        changed = true;
    }
  }

  return changed;
}

} // namespace luci
