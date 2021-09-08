
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

#include "luci/Pass/SubstituteSplitVToSplitPass.h"

#include <loco.h>

#include <luci/IR/CircleNodes.h>
#include <luci/Profile/CircleNodeOrigin.h>

namespace
{

void copy_quantparam(luci::CircleNode *dst, const luci::CircleNode *src)
{
  auto q = src->quantparam();
  if (q == nullptr)
    dst->quantparam(nullptr);
  else
    dst->quantparam(std::make_unique<luci::CircleQuantParam>(*q));
}

bool resolve_splitv(luci::CircleSplitV *sv)
{
  auto size_splits = dynamic_cast<luci::CircleConst *>(sv->size_splits());
  if (not size_splits)
    return false;

  if (size_splits->dtype() != loco::DataType::S32)
    return false;

  auto num_split = size_splits->size<loco::DataType::S32>();
  if (static_cast<int32_t>(num_split) != sv->num_split())
    return false;

  if (num_split < 1)
    return false;

  // Check the contents of size_splits are all same
  auto first_size = size_splits->at<loco::DataType::S32>(0);
  for (uint32_t i = 1; i < num_split; i++)
  {
    if (first_size != size_splits->at<loco::DataType::S32>(i))
      return false;
  }

  auto graph = sv->graph();
  auto split_node = graph->nodes()->create<luci::CircleSplit>();
  split_node->input(sv->input());
  split_node->split_dim(sv->split_dim());
  split_node->num_split(sv->num_split());
  split_node->name(sv->name());
  copy_quantparam(split_node, sv);
  luci::add_origin(split_node, luci::get_origin(sv));

  auto succs = loco::succs(sv);
  for (auto succ : succs)
  {
    auto svo = loco::must_cast<luci::CircleSplitVOut *>(succ);
    auto so_node = graph->nodes()->create<luci::CircleSplitOut>();
    so_node->input(split_node);
    so_node->index(svo->index());
    so_node->name(svo->name());
    copy_quantparam(so_node, svo);
    luci::add_origin(so_node, luci::get_origin(svo));

    replace(svo).with(so_node);
  }

  return true;
}

} // namespace

namespace luci
{

/**
 *  EXAMPLE (SplitV with num_split = 2)
 *
 *  BEFORE
 *              [CircleNode]
 *                   |
 *             [CircleSplitV] (size_splits and split_dim are ignored)
 *                /      \
 *   [CircleSplitVOut]  [CircleSplitVOut]
 *            |                 |
 *       [CircleNode]     [CircleNode]
 *
 *  AFTER
 *                    [CircleNode]
 *                     /         \
 *             [CircleSplit]    [CircleSplitV] (dead)
 *                /      \               \
 *   [CircleSplitOut]  [CircleSplitOut]  [CircleSplitVOut] * 2 (dead)
 *            |                 |
 *       [CircleNode]     [CircleNode]
 */
bool SubstituteSplitVToSplitPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto sv = dynamic_cast<luci::CircleSplitV *>(node))
    {
      if (resolve_splitv(sv))
        changed = true;
    }
  }

  return changed;
}

} // namespace luci
