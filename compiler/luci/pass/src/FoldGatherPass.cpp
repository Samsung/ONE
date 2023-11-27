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

#include "luci/Pass/FoldGatherPass.h"

#include <luci/IR/CircleNodes.h>

namespace
{

/**
 * Fold to const if
 *
 * 1. params: const and dtype = S32 or S64
 * 2. indices: const and dtype = S32 or S64
 *
 * BEFORE
 *
 *    [CircleConst]              [CircleConst]
 *         |                          |
 *         +---------[Gather]---------+
 *
 * AFTER
 *
 *                [CircleConst]
 *
 **/
template <loco::DataType InputT, loco::DataType IndexT>
bool fold_gather(luci::CircleGather *gather_node)
{
  const auto params = loco::must_cast<luci::CircleConst *>(gather_node->params());
  const auto indices = loco::must_cast<luci::CircleConst *>(gather_node->indices());

  const auto rank = params->rank();
  auto axis = gather_node->axis();
  if (axis < 0)
  {
    axis += static_cast<int32_t>(rank);
  }

  if (axis < 0 or axis >= static_cast<int32_t>(rank))
    throw std::runtime_error("Unsupported axis value");

  const auto name = gather_node->name();
  assert(name.length() > 0);

  auto constant = gather_node->graph()->nodes()->create<luci::CircleConst>();
  constant->dtype(InputT);
  constant->name(name + "_folded");

  constant->rank(rank + indices->rank() - 1);

  assert(constant->rank() > 0);

  std::vector<uint32_t> shape;
  for (uint32_t i = 0; i < rank; ++i)
  {
    if (i != static_cast<uint32_t>(axis))
    {
      const auto dim = params->dim(i).value();
      shape.push_back(dim);
    }
    else
    {
      for (uint32_t j = 0; j < indices->rank(); ++j)
      {
        const auto dim = indices->dim(j).value();
        shape.push_back(dim);
      }
    }
  }

  uint32_t size = 1;
  for (uint32_t i = 0; i < shape.size(); ++i)
  {
    constant->dim(i).set(shape.at(i));
    size *= shape.at(i);
  }

  constant->size<InputT>(size);

  uint32_t outer_size = 1;
  for (uint32_t i = 0; i < static_cast<uint32_t>(axis); ++i)
  {
    outer_size *= params->dim(i).value();
  }

  uint32_t inner_size = 1;
  for (uint32_t i = axis + 1; i < rank; ++i)
  {
    inner_size *= params->dim(i).value();
  }

  uint32_t coord_size = 1;
  for (uint32_t i = 0; i < indices->rank(); ++i)
  {
    coord_size *= indices->dim(i).value();
  }

  const auto axis_size = params->dim(axis).value();

  for (uint32_t outer = 0; outer < outer_size; ++outer)
  {
    for (uint32_t i = 0; i < coord_size; ++i)
    {
      constant->at<InputT>((outer * coord_size + i) * inner_size) =
        params->at<InputT>((outer * axis_size + indices->at<IndexT>(i)) * inner_size);
    }
  }
  loco::replace(gather_node).with(constant);

  return true;
}

bool fold_gather(luci::CircleGather *gather_node)
{
  const auto params = dynamic_cast<luci::CircleConst *>(gather_node->params());
  if (not params)
    return false;

  const auto indices = dynamic_cast<luci::CircleConst *>(gather_node->indices());
  if (not indices)
    return false;

  // TODO: support more types
  if (params->dtype() != loco::DataType::S32 and params->dtype() != loco::DataType::S64)
    return false;

  if (indices->dtype() != loco::DataType::S32 and indices->dtype() != loco::DataType::S64)
    throw std::runtime_error("Unsupported type");

  if (params->dtype() == loco::DataType::S64)
  {
    if (indices->dtype() == loco::DataType::S64)
      return fold_gather<loco::DataType::S64, loco::DataType::S64>(gather_node);
    else
      return fold_gather<loco::DataType::S64, loco::DataType::S32>(gather_node);
  }
  else
  {
    if (indices->dtype() == loco::DataType::S64)
      return fold_gather<loco::DataType::S32, loco::DataType::S64>(gather_node);
    else
      return fold_gather<loco::DataType::S32, loco::DataType::S32>(gather_node);
  }
}

} // namespace

namespace luci
{

/**
 * Constant Folding for Gather Op
 **/
bool FoldGatherPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto gather_node = dynamic_cast<luci::CircleGather *>(node))
    {
      if (fold_gather(gather_node))
        changed = true;
    }
  }

  return changed;
}

} // namespace luci
