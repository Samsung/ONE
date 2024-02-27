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

#include <luci/Profile/CircleNodeOrigin.h>

#include "WeightDivider.h"

namespace
{

bool divideConst(luci::CircleConst *node)
{
  if (node->dtype() != loco::DataType::FLOAT32)
    return true;

  auto new_const_node = node->graph()->nodes()->create<luci::CircleConst>();
  new_const_node->dtype(node->dtype());

  new_const_node->size<loco::DataType::FLOAT32>(0);
  new_const_node->shape_status(luci::ShapeStatus::VALID);
  new_const_node->rank(node->rank());
  for (uint32_t i = 0; i < node->rank(); ++i)
  {
    new_const_node->dim(i) = node->dim(i);
  }

  new_const_node->name(node->name());
  luci::add_origin(new_const_node, luci::get_origin(node));

  replace(node).with(new_const_node);
  return true;
}

} // namespace

namespace luci
{

/**
 * Constant dividing
 **/
bool WeightDivider::divide()
{
  for (size_t i = 0; i < _module->size(); ++i)
  {
    loco::Graph *graph = _module->graph(i);
    for (auto node : loco::active_nodes(loco::output_nodes(graph)))
    {
      if (auto const_node = dynamic_cast<luci::CircleConst *>(node))
      {
        divideConst(const_node);
        //  return false;
      }
    }
  }

  return true;
}

} // namespace luci
