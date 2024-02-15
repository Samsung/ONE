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

#include "luci/Pass/RemoveGatherGuardPass.h"

#include <luci/IR/CircleNodes.h>

#include <climits>

namespace
{

/*
 * BEFORE
 *
 *   [CircleNode]  [CircleNode]
 *         |            |
 *         |       [CircleAdd]
 *         |            |
 *         |     [CircleFloorMod]
 *         |        /
 *      [CircleGather]
 *            |
 *       [CircleNode]
 *
 * AFTER
 *
 *   [CircleNode]  [CircleNode]
 *         |          |    \
 *         |          |   [CircleAdd]
 *         |          /        |
 *         |         /    [CircleFloorMod]
 *         |        /
 *      [CircleGather]
 *            |
 *       [CircleNode]
 */

bool is_single_value_equal(const loco::Node *node, int32_t value)
{
  assert(node);

  auto const cnode = dynamic_cast<const luci::CircleConst *>(node);
  if (cnode == nullptr)
    return false;
  if (not(cnode->rank() == 0 || (cnode->rank() == 1 && cnode->dim(0).value() == 1)))
    return false;

  if (cnode->dtype() == loco::DataType::S32)
    return cnode->at<loco::DataType::S32>(0) == value;
  else if (cnode->dtype() == loco::DataType::S64)
    return cnode->at<loco::DataType::S64>(0) == static_cast<int64_t>(value);

  return false;
}

bool remove_guards(luci::CircleGather *gather)
{
  assert(gather);
  // check if sequence is Add+FloorMod
  auto floormod = dynamic_cast<luci::CircleFloorMod *>(gather->indices());
  if (floormod == nullptr)
    return false;
  auto add = dynamic_cast<luci::CircleAdd *>(floormod->x());
  if (add == nullptr)
    return false;

  if (add->fusedActivationFunction() != luci::FusedActFunc::NONE)
    return false;

  // check if gather axis is 0 for now
  // TODO support other axis
  if (gather->axis() != 0)
    return false;
  // check if RHS of Add and FloorMod is Const and is scalar/single element and
  // the value is same as gather.params.dim(0)
  luci::CircleNode *params = loco::must_cast<luci::CircleNode *>(gather->params());
  if (params->shape_status() != luci::ShapeStatus::VALID || params->rank() == 0)
    return false;
  // safe range check
  if (params->dim(gather->axis()).value() >= INT_MAX)
    return false;
  int32_t params_axis_dim = static_cast<int32_t>(params->dim(gather->axis()).value());
  if (not is_single_value_equal(add->y(), params_axis_dim))
    return false;
  if (not is_single_value_equal(floormod->y(), params_axis_dim))
    return false;

  // disconnect Add+FloorMod
  gather->indices(add->x());

  return true;
}

} // namespace

namespace luci
{

bool RemoveGatherGuardPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto gather = dynamic_cast<luci::CircleGather *>(node))
    {
      if (remove_guards(gather))
        changed = true;
    }
  }
  return changed;
}

} // namespace luci
