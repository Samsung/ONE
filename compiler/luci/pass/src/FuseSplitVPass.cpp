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

#include "luci/Pass/FuseSplitVPass.h"

#include <luci/Service/CircleNodeClone.h>
#include <luci/IR/CircleNode.h>
#include <luci/Profile/CircleNodeOrigin.h>

#define RETURN_UNLESS(cond) \
  if (not(cond))            \
    return false;

namespace
{

/**
 * This pass fuses nested SplitV operations into one operation.
 * More formally, the pass transforms the following pattern
 *
 *                 [In]
 *                   |
 *                   V
 *    +----------- SplitV -----------+
 *    |                              |
 *    V                              V
 * SplitVOut                       SplitVOut
 *                                   |
 *                                   V
 *                    +----------- SplitV -----------+
 *                    |                              |
 *                    V                              V
 *                 SplitVOut                       SplitVOut
 *
 * Into
 *
 *                 [In]
 *                   |
 *                   V
 *    +----------- SplitV -----------+
 *    |              |               |
 *    V              V               V
 * SplitVOut      SplitVOut       SplitVOut
 */

/**
 * @brief Create SplitV node
 *
 * NOTE: This helper does NOT set:
 *  - input
 *  - shape
 *  - origin
 *  - name
 * of the resulting SplitV
 */
luci::CircleSplitV *create_split_v(loco::Graph *graph, uint32_t split_dim,
                                   const std::vector<uint32_t> &size_splits)
{
  auto split = graph->nodes()->create<luci::CircleSplitV>();
  split->num_split(size_splits.size());

  auto split_dim_node = graph->nodes()->create<luci::CircleConst>();
  split_dim_node->dtype(loco::DataType::S32);
  split_dim_node->size<loco::DataType::S32>(1);
  split_dim_node->name("split_dim");
  split_dim_node->rank(0);
  split_dim_node->at<loco::DataType::S32>(0) = split_dim;
  split_dim_node->shape_status(luci::ShapeStatus::VALID);

  split->split_dim(split_dim_node);

  auto size_splits_node = graph->nodes()->create<luci::CircleConst>();
  size_splits_node->dtype(loco::DataType::S32);
  size_splits_node->size<loco::DataType::S32>(size_splits.size());
  size_splits_node->name("size_splits");
  size_splits_node->rank(1);
  size_splits_node->shape({static_cast<uint32_t>(size_splits.size())});
  size_splits_node->shape_status(luci::ShapeStatus::VALID);

  for (uint32_t i = 0; i < size_splits.size(); ++i)
  {
    size_splits_node->at<loco::DataType::S32>(i) = size_splits.at(i);
  }

  split->size_splits(size_splits_node);

  return split;
}

bool fuse_splitv(luci::CircleSplitV *child_splitv)
{
  auto parent_vout = dynamic_cast<luci::CircleSplitVOut *>(child_splitv->input());
  RETURN_UNLESS(parent_vout);

  auto const child_split_dim = dynamic_cast<luci::CircleConst *>(child_splitv->split_dim());
  RETURN_UNLESS(child_split_dim);

  auto const idx = static_cast<uint32_t>(parent_vout->index());

  auto parent_splitv = dynamic_cast<luci::CircleSplitV *>(parent_vout->input());
  RETURN_UNLESS(parent_splitv);

  auto const parent_split_dim = dynamic_cast<luci::CircleConst *>(parent_splitv->split_dim());
  RETURN_UNLESS(parent_split_dim);
  RETURN_UNLESS(parent_split_dim->scalar<loco::DataType::S32>() ==
                child_split_dim->scalar<loco::DataType::S32>());

  uint32_t const split_dim = parent_split_dim->scalar<loco::DataType::S32>();

  auto parent_size_splits = dynamic_cast<luci::CircleConst *>(parent_splitv->size_splits());
  RETURN_UNLESS(parent_size_splits);

  auto child_size_splits = dynamic_cast<luci::CircleConst *>(child_splitv->size_splits());
  RETURN_UNLESS(child_size_splits);

  auto graph = child_splitv->graph();

  std::vector<uint32_t> fused_size_splits;
  for (uint32_t p = 0; p < idx; ++p)
  {
    fused_size_splits.emplace_back(parent_size_splits->at<loco::DataType::S32>(p));
  }

  for (uint32_t c = 0; c < child_size_splits->size<loco::DataType::S32>(); ++c)
  {
    fused_size_splits.emplace_back(child_size_splits->at<loco::DataType::S32>(c));
  }

  for (uint32_t p = idx + 1; p < parent_size_splits->size<loco::DataType::S32>(); ++p)
  {
    fused_size_splits.emplace_back(parent_size_splits->at<loco::DataType::S32>(p));
  }

  auto fused_split = create_split_v(graph, split_dim, fused_size_splits);
  {
    fused_split->input(parent_splitv->input());
    fused_split->rank(parent_splitv->rank());
    fused_split->dtype(loco::DataType::FLOAT32);
    for (uint32_t dim = 0; dim < fused_split->rank(); ++dim)
    {
      fused_split->dim(dim).set(parent_splitv->dim(dim).value());
    }
    fused_split->shape_status(luci::ShapeStatus::VALID);
    fused_split->name(parent_splitv->name() + "+" + child_splitv->name());
    luci::add_origin(fused_split, luci::composite_origin({luci::get_origin(parent_splitv),
                                                          luci::get_origin(child_splitv)}));
  }

  for (auto old_parent_svo_node : loco::succs(parent_splitv))
  {
    auto old_parent_svo = dynamic_cast<luci::CircleSplitVOut *>(old_parent_svo_node);
    RETURN_UNLESS(old_parent_svo);

    if (static_cast<uint32_t>(old_parent_svo->index()) < idx)
    {
      auto clone_svo =
        loco::must_cast<luci::CircleSplitVOut *>(luci::clone_node(old_parent_svo, graph));
      clone_svo->input(fused_split);
      clone_svo->index(old_parent_svo->index());
      loco::replace(old_parent_svo).with(clone_svo);
    }
    else if (static_cast<uint32_t>(old_parent_svo->index()) == idx)
    {
      for (auto old_child_svo_node : loco::succs(child_splitv))
      {
        auto old_child_svo = dynamic_cast<luci::CircleSplitVOut *>(old_child_svo_node);
        RETURN_UNLESS(old_child_svo);
        auto clone_svo =
          loco::must_cast<luci::CircleSplitVOut *>(luci::clone_node(old_child_svo, graph));
        clone_svo->index(old_child_svo->index() + idx);
        clone_svo->input(fused_split);
        loco::replace(old_child_svo).with(clone_svo);
      }
    }
    else
    {
      auto clone_svo =
        loco::must_cast<luci::CircleSplitVOut *>(luci::clone_node(old_parent_svo, graph));
      clone_svo->input(fused_split);
      clone_svo->index(old_parent_svo->index() + child_size_splits->size<loco::DataType::S32>() -
                       1);
      loco::replace(old_parent_svo).with(clone_svo);
    }
  }

  return true;
}
} // namespace

namespace luci
{

bool FuseSplitVPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto splitv = dynamic_cast<luci::CircleSplitV *>(node);
    if (not splitv)
      continue;

    if (fuse_splitv(splitv))
      changed = true;
  }

  return changed;
}

} // namespace luci
