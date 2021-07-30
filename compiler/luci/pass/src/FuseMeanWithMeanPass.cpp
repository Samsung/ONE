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

#include "luci/Pass/FuseMeanWithMeanPass.h"

#include <luci/IR/CircleNode.h>
#include <luci/Profile/CircleNodeOrigin.h>

namespace
{
/**
 *  Fuse two Mean op to one Mean op with merged reduction indices
 *
 *  BEFORE
 *                  |
 *          [CircleMean, axis<1>]
 *                  |
 *         [CircleMean, axis<1>]
 *                  |
 *
 *  AFTER
 *                  |
 *          [CircleMean, axis<1,2>]
 *                  |
 *
 */
luci::CircleConst *create_fused_indices(luci::CircleConst *indices, int32_t index_value,
                                        int32_t next_index_value)
{
  auto name = indices->name();
  assert(name.length() > 0);

  auto fused_indices_const = indices->graph()->nodes()->create<luci::CircleConst>();
  fused_indices_const->dtype(indices->dtype());
  fused_indices_const->rank(1);
  fused_indices_const->size<loco::DataType::S32>(2);
  fused_indices_const->at<loco::DataType::S32>(0) = index_value;
  fused_indices_const->at<loco::DataType::S32>(1) = next_index_value;
  fused_indices_const->shape_status(luci::ShapeStatus::VALID);
  fused_indices_const->name(name);

  return fused_indices_const;
}

bool fuse_mean_with_mean(luci::CircleMean *mean)
{
  // Check whether input rank of current CircleMean operation is less 2 or not.
  // This optimization works only if doesn't.
  auto input = loco::must_cast<luci::CircleNode *>(mean->input());
  if (input->shape_status() != luci::ShapeStatus::VALID)
    return false;
  if (input->rank() < 2)
    return false;

  // Get reduction indices of current CircleMean operation.
  auto indices = dynamic_cast<luci::CircleConst *>(mean->reduction_indices());
  if (not indices)
    return false;
  assert(indices->dtype() == loco::DataType::S32);

  // Check whether indices size is equal to 1 ot not.
  auto indices_size = indices->size<loco::DataType::S32>();
  if (indices_size != 1)
    return false;

  // Get index value of current CircleMean operation.
  auto index_value = indices->at<loco::DataType::S32>(0);

  // Get next node after current CircleMean operation.
  auto next_nodes = loco::succs(mean);
  if (next_nodes.size() != 1)
    return false;

  // Check whether next node is CircleMean operation or not.
  auto next_mean = dynamic_cast<luci::CircleMean *>(*next_nodes.begin());
  if (not next_mean)
    return false;

  // Do the same checks as before for next_mean CircleMean operation.
  auto next_indices = dynamic_cast<luci::CircleConst *>(next_mean->reduction_indices());
  if (not next_indices)
    return false;
  assert(next_indices->dtype() == loco::DataType::S32);

  auto next_indices_size = next_indices->size<loco::DataType::S32>();
  if (next_indices_size != 1)
    return false;

  auto next_index_value = next_indices->at<loco::DataType::S32>(0);

  // Before merge indices of this two CircleMean operations correct next index value.
  if (index_value <= next_index_value and (not mean->keep_dims()))
  {
    next_index_value += 1;
  }

  // Create merged indices.
  auto fused_indices_const = create_fused_indices(indices, index_value, next_index_value);

  auto name = mean->name();
  assert(name.length() > 0);

  // Create and configure new CircleMean operation.
  auto fused_mean = mean->graph()->nodes()->create<luci::CircleMean>();
  fused_mean->reduction_indices(fused_indices_const);
  fused_mean->input(mean->input());
  fused_mean->keep_dims(next_mean->keep_dims());
  fused_mean->name(name);

  // Replace old CircleMeans operations with new CircleMean operation with merged indices.
  replace(next_mean).with(fused_mean);
  luci::add_origin(fused_mean, luci::get_origin(next_mean));

  return true;
}

} // namespace

namespace luci
{

bool FuseMeanWithMeanPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto mean = dynamic_cast<luci::CircleMean *>(node);
    if (not mean)
      continue;

    if (fuse_mean_with_mean(mean))
      changed = true;
  }

  return changed;
}

} // namespace luci
