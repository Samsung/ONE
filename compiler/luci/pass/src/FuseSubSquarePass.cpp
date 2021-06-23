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

#include "luci/Pass/FuseSubSquarePass.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Profile/CircleNodeOrigin.h>

namespace
{
/**
 *  Fuse Sub-Square sequence to SquaredDifference
 *
 *  BEFORE
 *
 *   [CircleNode]  [CircleNode]
 *            |      |
 *           [CircleSub]
 *                |
 *         [CircleSquare]
 *                |
 *          [CircleNode]
 *
 *  AFTER
 *
 *   [CircleNode]  [CircleNode]
 *            |      |
 *    [CircleSquaredDifference]   [CircleSub]
 *               |                     |
 *               |               [CircleSquare]
 *          [CircleNode]
 */
bool fuse_sub_square(luci::CircleSquare *square)
{
  // check whether it has bias or not. This optimization works only if it doesn't.
  auto sub = dynamic_cast<luci::CircleSub *>(square->x());
  if (not sub)
    return false;

  if (sub->fusedActivationFunction() != luci::FusedActFunc::NONE)
    return false;

  auto squdiff = square->graph()->nodes()->create<luci::CircleSquaredDifference>();
  squdiff->x(sub->x());
  squdiff->y(sub->y());
  squdiff->name(sub->name() + ";" + square->name());
  squdiff->dtype(sub->dtype());
  squdiff->rank(sub->rank());
  for (uint32_t i = 0; i < sub->rank(); i++)
    squdiff->dim(i).set(sub->dim(i).value());

  replace(square).with(squdiff);

  luci::add_origin(squdiff, luci::get_origin(sub));
  luci::add_origin(squdiff, luci::get_origin(square));

  return true;
}

} // namespace

namespace luci
{

bool FuseSubSquarePass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto square = dynamic_cast<luci::CircleSquare *>(node);
    if (not square)
      continue;

    if (fuse_sub_square(square))
      changed = true;
  }

  return changed;
}

} // namespace luci
