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

#include "luci/Pass/ResolveCustomOpSplitVPass.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Profile/CircleNodeOrigin.h>
#include <luci/Service/Nodes/CircleConst.h>

namespace
{

// Input node is const S64
// Return s32 version of node
// Return nullptr if s64 value is out of range of s32
luci::CircleConst *s64_to_s32(luci::CircleConst *node)
{
  assert(node);
  assert(node->dtype() == loco::DataType::S64);

  auto cloned = luci::clone(node);
  luci::add_origin(cloned, luci::get_origin(node));

  const auto num_elems = node->size<loco::DataType::S64>();

  cloned->dtype(loco::DataType::S32);
  cloned->size<loco::DataType::S32>(num_elems);

  for (uint32_t i = 0; i < num_elems; i++)
  {
    int64_t val = node->at<loco::DataType::S64>(i);
    if (val < std::numeric_limits<int32_t>::min() or val > std::numeric_limits<int32_t>::max())
      return nullptr;

    cloned->at<loco::DataType::S32>(i) = static_cast<int32_t>(val);
  }

  return cloned;
}

/** BEFORE
 *
 *        [CircleNode]
 *              \
 *               \   [size_splits]  [split_dim]
 *                \       |             /
 *               [SplitV(CircleCustom)]
 *                        |
 *                 [CircleCustomOut]
 *                        |
 *
 *  AFTER
 *
 *        [CircleNode]
 *              \
 *               \   [size_splits]  [split_dim]
 *                \       |         /
 *                 \      |       /
 *                  \     |      /
 *                     [SplitV]
 *                        |
 *                    [SplitVOut]
 *                        |
 */
bool resolve_splitv(luci::CircleCustom *node)
{
  const std::string custom_code = node->custom_code();
  const std::vector<uint8_t> custom_options = node->custom_options();

  if (custom_code != "SplitV")
    return false;

  if (node->numInputs() != 3)
    return false;

  auto size_splits = dynamic_cast<luci::CircleConst *>(node->inputs(1));
  if (not size_splits)
    return false;

  // Convert size_splits to S32, because luci-interpeter does not support
  // S64 size_splits yet
  // TODO Support S64 size_splits
  if (size_splits->dtype() == loco::DataType::S64)
  {
    size_splits = s64_to_s32(size_splits);
    if (not size_splits)
      return false;
  }
  if (size_splits->dtype() != loco::DataType::S32)
    return false;

  auto split_dim = dynamic_cast<luci::CircleConst *>(node->inputs(2));
  if (not split_dim)
    return false;

  if (split_dim->dtype() == loco::DataType::S64)
  {
    split_dim = s64_to_s32(split_dim);
    if (not split_dim)
      return false;
  }
  if (split_dim->dtype() != loco::DataType::S32)
    return false;

  if (size_splits->rank() != 1)
    return false;

  const auto num_split = size_splits->dim(0).value();

  auto split_v = node->graph()->nodes()->create<luci::CircleSplitV>();
  split_v->input(node->inputs(0));
  split_v->size_splits(size_splits);
  split_v->split_dim(split_dim);
  split_v->num_split(num_split);
  split_v->name(node->name());
  luci::add_origin(split_v, luci::get_origin(node));

  int32_t i = 0;
  const auto succs = loco::succs(node);
  for (auto succ : succs)
  {
    auto custom_out = loco::must_cast<luci::CircleCustomOut *>(succ); // FIX_CALLER_UNLESS

    auto split_v_out = node->graph()->nodes()->create<luci::CircleSplitVOut>();
    split_v_out->input(split_v);
    split_v_out->name(node->name() + "_out_" + std::to_string(i));
    split_v_out->index(i++);
    luci::add_origin(split_v_out, luci::get_origin(node));
    loco::replace(custom_out).with(split_v_out);
  }

  return true;
}

} // namespace

namespace luci
{

bool ResolveCustomOpSplitVPass::run(loco::Graph *g)
{
  bool changed = false;

  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto cop = dynamic_cast<luci::CircleCustom *>(node);
    if (not cop)
      continue;

    if (resolve_splitv(cop))
      changed = true;
  }

  return changed;
}

} // namespace luci
