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

#include "luci/Pass/FuseTransposeWithMeanPass.h"

#include <luci/IR/CircleNode.h>
#include <luci/Profile/CircleNodeOrigin.h>
#include <luci/Service/Nodes/CircleConst.h>

namespace
{

/**
 *  Create a const for fused reduction indices
 *
 *  BEFORE
 *                  |
 *          [CircleTranspose, perm<0, 2, 3, 1>]
 *                  |
 *          [CircleMean, axis<3>]
 *                  |
 *
 *  AFTER
 *                  |
 *          [CircleMean, axis<1>]       [CircleTranspose, perm<0, 2, 3, 1>]
 *                  |                            |
 *                                      [CircleMean, axis<3>]
 *
 */
luci::CircleConst *create_fused_indices(luci::CircleConst *rindices,
                                        const std::vector<uint32_t> &fused_rindices)
{
  assert(rindices != nullptr); // FIX_CALLER_UNLESS

  if (rindices->dtype() != loco::DataType::S32)
    return nullptr;

  assert(fused_rindices.size() == rindices->size<loco::DataType::S32>());

  auto fused_rindices_const = luci::clone(rindices);
  auto name = rindices->name();
  assert(name.length() > 0); // FIX_CALLER_UNLESS
  fused_rindices_const->name(name + "_orig");

  for (uint32_t i = 0; i < fused_rindices.size(); ++i)
  {
    fused_rindices_const->at<loco::DataType::S32>(i) = fused_rindices.at(i);
  }

  return fused_rindices_const;
}

bool const_has_value_s32(const luci::CircleConst *circle_const, int32_t value)
{
  if (circle_const->dtype() != loco::DataType::S32)
    return false;

  uint32_t size = circle_const->size<loco::DataType::S32>();
  for (uint32_t i = 0; i < size; ++i)
  {
    if (circle_const->at<loco::DataType::S32>(i) == value)
      return true;
  }

  return false;
}

bool fuse_transpose_with_mean(luci::CircleMean *mean)
{
  auto transpose = dynamic_cast<luci::CircleTranspose *>(mean->input());
  if (not transpose)
    return false;

  // Get reduction indices of CircleMean operation.
  auto rindices = dynamic_cast<luci::CircleConst *>(mean->reduction_indices());
  if (not rindices)
    return false;

  if (rindices->dtype() != loco::DataType::S32)
    return false;

  if (mean->keep_dims() != false)
    return false;

  auto perm = dynamic_cast<luci::CircleConst *>(transpose->perm());
  if (not perm)
    return false;

  std::vector<uint32_t> axes_after_reduction;
  std::vector<uint32_t> orig_reduced_axes;
  for (uint32_t axis = 0; axis < perm->size<loco::DataType::S32>(); ++axis)
  {
    uint32_t original_axis = static_cast<uint32_t>(perm->at<loco::DataType::S32>(axis));

    if (const_has_value_s32(rindices, axis))
    {
      orig_reduced_axes.push_back(original_axis);
      continue;
    }

    axes_after_reduction.push_back(original_axis);
  }

  if (not std::is_sorted(axes_after_reduction.begin(), axes_after_reduction.end()))
    return false;

  auto fused_rindices = create_fused_indices(rindices, orig_reduced_axes);
  if (not fused_rindices)
    return false;

  // Create and configure new CircleMean operation.
  auto fused_mean = mean->graph()->nodes()->create<luci::CircleMean>();
  fused_mean->reduction_indices(fused_rindices);
  fused_mean->input(transpose->a());
  fused_mean->keep_dims(false);
  fused_mean->name(mean->name() + "/Transpose");

  // Replace old CircleMean operations with new CircleMean operation with merged indices.
  replace(mean).with(fused_mean);
  luci::add_origin(fused_mean,
                   luci::composite_origin({luci::get_origin(mean), luci::get_origin(transpose)}));

  return true;
}

} // namespace

namespace luci
{

bool FuseTransposeWithMeanPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto mean = dynamic_cast<luci::CircleMean *>(node);
    if (not mean)
      continue;

    if (fuse_transpose_with_mean(mean))
      changed = true;
  }

  return changed;
}

} // namespace luci
