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

#include "luci/Pass/FoldSqueezePass.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Profile/CircleNodeOrigin.h>
#include <luci/Service/Nodes/CircleConst.h>

namespace
{

/**
 * Fold Squeeze to const if it has const input
 **/
bool fold_squeeze(luci::CircleSqueeze *squeeze)
{
  // Check squeeze has const input
  auto const_input = dynamic_cast<luci::CircleConst *>(squeeze->input());
  if (not const_input)
    return false;

  // Check all dimensions are known
  const auto input_rank = const_input->rank();
  for (uint32_t i = 0; i < input_rank; i++)
  {
    if (not const_input->dim(i).known())
      return false;
  }

  const auto squeeze_dims = squeeze->squeeze_dims();
  uint32_t num_squeeze_dims = squeeze_dims.size();
  std::vector<bool> should_squeeze(input_rank, false);
  uint32_t num_squeezed_dims = 0;

  // Squeeze all dimensions whose value is 1
  if (num_squeeze_dims == 0)
  {
    for (uint32_t idx = 0; idx < input_rank; ++idx)
    {
      if (const_input->dim(idx).value() == 1)
      {
        should_squeeze.at(idx) = true;
        ++num_squeezed_dims;
      }
    }
  }
  else
  {
    for (uint32_t idx = 0; idx < num_squeeze_dims; ++idx)
    {
      const int32_t current =
        squeeze_dims.at(idx) < 0 ? squeeze_dims.at(idx) + input_rank : squeeze_dims.at(idx);
      assert(current >= 0);
      assert(current < static_cast<int32_t>(input_rank));
      assert(const_input->dim(current).value() == 1);

      if (not should_squeeze[current])
        ++num_squeezed_dims;
      should_squeeze[current] = true;
    }
  }

  auto new_const = luci::clone(const_input);
  new_const->rank(input_rank - num_squeezed_dims);
  for (uint32_t in_idx = 0, out_idx = 0; in_idx < input_rank; ++in_idx)
  {
    if (should_squeeze.at(in_idx))
      continue;

    new_const->dim(out_idx++) = const_input->dim(in_idx);
  }

  new_const->shape_status(luci::ShapeStatus::VALID);

  new_const->name(const_input->name() + "_squeezed");
  luci::add_origin(
    new_const, luci::composite_origin({luci::get_origin(squeeze), luci::get_origin(const_input)}));

  loco::replace(squeeze).with(new_const);

  return true;
}

} // namespace

namespace luci
{

/**
 * Constant Folding for Squeeze Op
 **/
bool FoldSqueezePass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto squeeze = dynamic_cast<luci::CircleSqueeze *>(node))
    {
      if (fold_squeeze(squeeze))
        changed = true;
    }
  }

  return changed;
}

} // namespace luci
