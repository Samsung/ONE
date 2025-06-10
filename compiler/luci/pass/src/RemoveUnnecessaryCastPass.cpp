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

#include "luci/Pass/RemoveUnnecessaryCastPass.h"

#include <luci/IR/CircleNodes.h>

namespace
{

#define RETURN_FALSE_UNLESS(cond) \
  if (not(cond))                  \
    return false;

/**
 * BEFORE
 *
 *      [CircleNode]
 *            |
 *            |
 *      [CircleCast] (in_data_type == out_data_type)
 *            |
 *            |
 *      [CircleNode]
 *
 * AFTER
 *
 *      [CircleNode]
 *            |
 *            |           [CircleCast]
 *            |
 *      [CircleNode]
 *
 **/
bool remove_unnecessary_cast(luci::CircleCast *cast)
{
  RETURN_FALSE_UNLESS(cast->in_data_type() == cast->out_data_type());

  loco::replace(cast).with(cast->x());

  return true;
}

} // namespace

namespace luci
{

bool RemoveUnnecessaryCastPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto cast_node = dynamic_cast<luci::CircleCast *>(node))
    {
      if (remove_unnecessary_cast(cast_node))
        changed = true;
    }
  }
  return changed;
}

} // namespace luci
