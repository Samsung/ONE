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

#include "luci/Pass/RemoveUnnecessaryReshapePass.h"

#include <luci/IR/CircleNodes.h>

namespace
{

bool remove_no_effect_reshape(luci::CircleNode *node)
{
  auto target_node = dynamic_cast<luci::CircleReshape *>(node);
  if (target_node == nullptr)
    return false;

  std::vector<int32_t> shape_info;
  auto new_shape = dynamic_cast<luci::CircleConst *>(target_node->shape());
  if (new_shape == nullptr)
  {
    auto dummy_shape = dynamic_cast<luci::CircleOutputDummy *>(target_node->shape());
    if (dummy_shape == nullptr)
      return false;

    // Need to extract new_shape info on target_node->newShape()
    shape_info.resize(target_node->newShape()->rank());
    for (uint32_t i = 0; i < target_node->newShape()->rank(); i++)
      shape_info.at(i) = target_node->newShape()->dim(i);
  }
  else
  {
    shape_info.resize(new_shape->dim(0).value());
    for (uint32_t i = 0; i < new_shape->dim(0).value(); i++)
      shape_info.at(i) = new_shape->at<loco::DataType::S32>(i);
  }

  auto input_node = loco::must_cast<luci::CircleNode *>(target_node->tensor());
  if (input_node->rank() != shape_info.size())
    return false;
  for (uint32_t i = 0; i < input_node->rank(); i++)
  {
    if (shape_info.at(i) == -1)
      continue;
    if (input_node->shape_signature().rank() > 0 && input_node->shape_signature().dim(i) == -1)
      return false;
    if (input_node->dim(i).value() != shape_info.at(i))
      return false;
  }

  replace(target_node).with(input_node);
  return true;
}

} // namespace

namespace luci
{

bool RemoveUnnecessaryReshapePass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    if (remove_no_effect_reshape(circle_node))
    {
      changed = true;
    }
  }
  return changed;
}

} // namespace luci
