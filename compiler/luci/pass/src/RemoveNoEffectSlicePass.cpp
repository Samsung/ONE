/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/RemoveNoEffectSlicePass.h"

#include <luci/IR/CircleNodes.h>

namespace
{

template <loco::DataType DT>
bool check_input_output_shape(luci::CircleNode *input, luci::CircleSlice *target,
                              luci::CircleConst *begin, luci::CircleConst *size)
{
  for (uint32_t i = 0; i < input->rank(); i++)
  {
    int64_t size_value = static_cast<int64_t>(size->at<DT>(i));
    if (size_value < 0)
    {
      if (size_value != -1)
        return false;
      size_value =
          static_cast<int64_t>(input->dim(i).value()) - static_cast<int64_t>(begin->at<DT>(i));
    }
    else
    {
      if (static_cast<int64_t>(input->dim(i).value()) <
          static_cast<int64_t>(begin->at<DT>(i)) + size_value)
        return false;
    }
    if (size_value != static_cast<int64_t>(input->dim(i).value()))
      return false;
  }
  return true;
}

bool remove_no_effect_slice(luci::CircleNode *node)
{
  auto target_node = dynamic_cast<luci::CircleSlice *>(node);
  if (target_node == nullptr)
    return false;
  auto begin_const = dynamic_cast<luci::CircleConst *>(target_node->begin());
  if (target_node == nullptr)
    return false;
  auto size_const = dynamic_cast<luci::CircleConst *>(target_node->size());
  if (size_const == nullptr)
    return false;
  // Check input output shape.
  auto input_node = loco::must_cast<luci::CircleNode *>(target_node->input());
  if (begin_const->dtype() == loco::DataType::S32)
  {
    if (!check_input_output_shape<loco::DataType::S32>(input_node, target_node, begin_const,
                                                       size_const))
      return false;
    replace(target_node).with(input_node);
  }
  else if (begin_const->dtype() == loco::DataType::S64)
  {
    if (!check_input_output_shape<loco::DataType::S64>(input_node, target_node, begin_const,
                                                       size_const))
      return false;
    replace(target_node).with(input_node);
  }
  else
    return false;
  return true;
}

} // namespace

namespace luci
{
/**
 *   BEFORE
 *      |
 * [CircleNode]
 *      |
 * [CircleSlice]
 *      |
 * [CircleNode](with same shape)
 *      |
 *
 *    AFTER
 *      |
 * [CircleNode] Remove Slice OP
 *      |
 */
bool RemoveNoEffectSlicePass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    if (remove_no_effect_slice(circle_node))
    {
      changed = true;
    }
  }
  return changed;
}

} // namespace luci
