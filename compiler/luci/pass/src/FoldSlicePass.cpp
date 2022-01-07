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

#include "luci/Pass/FoldSlicePass.h"

#include <luci/IR/CircleNodes.h>

namespace
{

/**
 *  Graph that has a Slce Op with constant input
 *
 *  BEFORE
 *
 *    [CircleConst] [CircleConst] [CircleConst]
 *                \       |       /
 *                  [CircleSlice]
 *                        |
 *  AFTER
 *
 *    [CircleConst]    [CircleConst] [CircleConst] [CircleConst]
 *          |                     \       |       /
 *          |                       [CircleSlice]
 */

/**
 * Fold Cast to const if it has const input
 **/
bool fold_slice(luci::CircleSlice *slice)
{
  // Check slice has const input
  auto const_input = dynamic_cast<luci::CircleConst *>(slice->input());
  if (not const_input)
    return false;
  auto const_begin = dynamic_cast<luci::CircleConst *>(slice->begin());
  if (not const_begin)
    return false;
  auto const_size = dynamic_cast<luci::CircleConst *>(slice->size());
  if (not const_size)
    return false;

  // TODO implement

  return false;
}

} // namespace

namespace luci
{

/**
 * Constant Folding for Slice Op
 **/
bool FoldSlicePass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto cast = dynamic_cast<luci::CircleSlice *>(node))
    {
      if (fold_slice(cast))
        changed = true;
    }
  }

  return changed;
}

} // namespace luci
