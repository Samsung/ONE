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

#include "luci/Pass/FoldReshapePass.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Profile/CircleNodeOrigin.h>
#include <luci/Service/Nodes/CircleConst.h>

namespace
{

/**
 * Fold Reshape to const if it has const input
 **/
bool fold_reshape(luci::CircleReshape *reshape)
{
  // Check const input
  auto const_input = dynamic_cast<luci::CircleConst *>(reshape->tensor());
  if (not const_input)
    return false;

  // Check const shape
  auto const_shape = dynamic_cast<luci::CircleConst *>(reshape->shape());
  if (not const_shape)
    return false;

  // Check all dimensions are known
  const auto input_rank = const_input->rank();
  for (uint32_t i = 0; i < input_rank; i++)
  {
    if (not const_input->dim(i).known())
      return false;
  }

  // Check all dimensions are known
  const auto shape_rank = const_shape->rank();
  if (shape_rank != 1)
    return false;

  if (not const_shape->dim(0).known())
    return false;

  std::vector<uint32_t> new_shape;
  switch (const_shape->dtype())
  {
    case loco::DataType::S32:
      for (uint32_t i = 0; i < const_shape->size<loco::DataType::S32>(); i++)
      {
        new_shape.push_back(const_shape->at<loco::DataType::S32>(i));
      }
      break;
    // TODO Support S64
    default:
      return false;
  }

  auto new_const = luci::clone(const_input);
  new_const->rank(new_shape.size());
  for (uint32_t i = 0; i < new_shape.size(); i++)
  {
    new_const->dim(i).set(new_shape[i]);
  }

  new_const->shape_status(luci::ShapeStatus::VALID);

  new_const->name(const_input->name() + "_reshaped");
  luci::add_origin(
    new_const, luci::composite_origin({luci::get_origin(reshape), luci::get_origin(const_input)}));

  loco::replace(reshape).with(new_const);

  return true;
}

} // namespace

namespace luci
{

/**
 * Constant Folding for Reshape Op
 **/
bool FoldReshapePass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto reshape = dynamic_cast<luci::CircleReshape *>(node))
    {
      if (fold_reshape(reshape))
        changed = true;
    }
  }

  return changed;
}

} // namespace luci
