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
 *  Fuse two Mean operations to one Mean operation with merged reduction indices
 *
 *  BEFORE
 *                  |
 *          [CircleMean, axis<1>]
 *                  |
 *          [CircleMean, axis<1>]
 *                  |
 *
 *  AFTER
 *                  |
 *          [CircleMean, axis<1,2>]     [CircleMean, axis<1>]
 *                  |                            |
 *                                      [CircleMean, axis<1>]
 *
 */
luci::CircleConst *create_fused_indices(luci::CircleConst *indices,
                                        const std::set<uint32_t> &indices_set)
{
  auto name = indices->name();

  auto fused_indices_const = indices->graph()->nodes()->create<luci::CircleConst>();
  fused_indices_const->dtype(indices->dtype());
  fused_indices_const->rank(1);
  fused_indices_const->size<loco::DataType::S32>(indices_set.size());
  fused_indices_const->shape_status(luci::ShapeStatus::VALID);
  fused_indices_const->name(name);

  auto curr_index = 0;
  for (auto it = indices_set.begin(); it != indices_set.end(); it++)
  {
    fused_indices_const->at<loco::DataType::S32>(curr_index) = *it;
    curr_index++;
  }

  return fused_indices_const;
}

bool fuse_mean_with_mean(luci::CircleMean *mean)
{
  // Get reduction indices of current CircleMean operation.
  auto indices = dynamic_cast<luci::CircleConst *>(mean->reduction_indices());
  if (not indices)
    return false;
  assert(indices->dtype() == loco::DataType::S32);

  // Check whether previous node is CircleMean operation or not.
  auto prev_mean = dynamic_cast<luci::CircleMean *>(mean->input());
  if (not prev_mean)
    return false;

  // Check whether input rank of previous CircleMean operation is less 2 or not.
  // This optimization works only if doesn't.
  auto input = loco::must_cast<luci::CircleNode *>(prev_mean->input());
  if (input->shape_status() != luci::ShapeStatus::VALID)
    return false;
  auto input_rank = input->rank();
  if (input_rank < 2)
    return false;

  // Check whether current CircleMean and next CircleMean
  // has the same keep_dims parameter or not.
  // If it doesn't, keep the graph unchanged.
  if (mean->keep_dims() != prev_mean->keep_dims())
    return false;

  // Get reduction indices of previous CircleMean operation.
  auto prev_indices = dynamic_cast<luci::CircleConst *>(prev_mean->reduction_indices());
  if (not prev_indices)
    return false;
  assert(prev_indices->dtype() == loco::DataType::S32);

  // Get sizes of indices of current CircleMean operation and previous CircleMean operation.
  auto indices_size = indices->size<loco::DataType::S32>();
  auto prev_indices_size = prev_indices->size<loco::DataType::S32>();

  // Get set of indices of current CircleMean operation.
  std::set<uint32_t> indices_set;
  for (uint32_t i = 0; i < prev_indices_size; i++)
  {
    auto index = prev_indices->at<loco::DataType::S32>(i);
    if (index < 0)
      index += input_rank;
    indices_set.insert(index);
  }

  // Get the vector of input indexes, that remained untouched
  // after the current CircleMean operation.
  std::vector<uint32_t> input_indices_vector;
  for (uint32_t i = 0; i < input_rank; i++)
  {
    if (indices_set.find(i) == indices_set.end())
      input_indices_vector.push_back(i);
  }

  // Get final set of merged indices.
  for (uint32_t i = 0; i < indices_size; i++)
  {
    auto index = indices->at<loco::DataType::S32>(i);
    if (index < 0)
      index += input_rank;
    indices_set.insert(input_indices_vector.at(index));
  }

  // Create merged indices.
  auto fused_indices_const = create_fused_indices(indices, indices_set);

  auto name = mean->name();
  assert(name.length() > 0);

  // Create and configure new CircleMean operation.
  auto fused_mean = mean->graph()->nodes()->create<luci::CircleMean>();
  fused_mean->reduction_indices(fused_indices_const);
  fused_mean->input(prev_mean->input());
  fused_mean->keep_dims(mean->keep_dims());
  fused_mean->name(name + "/Mean");

  // Replace old CircleMeans operations with new CircleMean operation with merged indices.
  replace(mean).with(fused_mean);
  luci::add_origin(fused_mean,
                   luci::composite_origin({luci::get_origin(mean), luci::get_origin(prev_mean)}));

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
