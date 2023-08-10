/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/DecomposeHardSwishPass.h"

#include "helpers/NodeFiller.h"
#include "helpers/TypeMapper.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Profile/CircleNodeOrigin.h>

namespace
{
/**
 *  BEFORE
 *        [CircleNode]
 *              |
 *              |
 *      [CircleHardSwish]
 *              |
 *              |
 *        [CircleNode]
 *
 *
 *  AFTER
 *
 *      [CircleNode]  [CircleConst]
 *          |    \       /
 *          |     \     /
 *          |   [CircleAdd]
 *          |        |
 *          |        |
 *          \  [CircleRelu6] [CircleConst]
 *           \        \        /
 *            \        \      /
 *             \      [CircleMul]
 *              \       /
 *               \     /
 *             [CircleMul]
 *                  |
 *                  |
 *             [CircleNode]
 *
 */
bool decompose_hardswish(luci::CircleHardSwish *hardswish)
{
  if (not hardswish)
    return false;

  if (hardswish->dtype() != loco::DataType::FLOAT32)
    return false;

  auto g = hardswish->graph();

  auto name = hardswish->name();
  assert(name.length() > 0);

  // Create a const for CircleAdd operation
  auto add_const = g->nodes()->create<luci::CircleConst>();
  add_const->shape({}); // scalar
  add_const->dtype(loco::DataType::FLOAT32);
  add_const->rank(0);
  add_const->size<loco::DataType::FLOAT32>(1);
  add_const->at<loco::DataType::FLOAT32>(0) = 3.;
  add_const->name("add_const");
  luci::add_origin(add_const, luci::get_origin(hardswish));

  // Create an Add operation
  auto add = g->nodes()->create<luci::CircleAdd>();
  add->fusedActivationFunction(luci::FusedActFunc::NONE); // TODO: Seungho
  add->x(hardswish->features());
  add->y(add_const);
  add->name(name + "/Add");
  luci::add_origin(add, luci::get_origin(hardswish));

  // Create a Relu6 operation
  auto relu6 = g->nodes()->create<luci::CircleRelu6>();
  relu6->features(add);
  relu6->name(name + "/Relu6");
  luci::add_origin(relu6, luci::get_origin(hardswish));

  // Create a const for CircleMul operation
  auto mul_const = g->nodes()->create<luci::CircleConst>();
  mul_const->shape({}); // scalar
  mul_const->dtype(loco::DataType::FLOAT32);
  mul_const->rank(0);
  mul_const->size<loco::DataType::FLOAT32>(1);
  mul_const->at<loco::DataType::FLOAT32>(0) = 1. / 6.;
  mul_const->name("mul_const");
  luci::add_origin(mul_const, luci::get_origin(hardswish));

  // Create first Mul operation
  auto mul1 = g->nodes()->create<luci::CircleMul>();
  mul1->fusedActivationFunction(luci::FusedActFunc::NONE);
  mul1->x(relu6);
  mul1->y(mul_const);
  mul1->name(name + "/Mul1");
  luci::add_origin(mul1, luci::get_origin(hardswish));

  // Create second Mul operation
  auto mul2 = g->nodes()->create<luci::CircleMul>();
  mul2->fusedActivationFunction(luci::FusedActFunc::NONE);
  mul2->x(hardswish->features());
  mul2->y(mul1);
  mul2->name(name + "/Mul2");
  luci::add_origin(mul2, luci::get_origin(hardswish));

  replace(hardswish).with(mul2);

  return true;
}

} // namespace

namespace luci
{

bool DecomposeHardSwishPass::run(loco::Graph *g)
{
  bool changed = false;

  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto hardswish = dynamic_cast<luci::CircleHardSwish *>(node))
    {
      if (decompose_hardswish(hardswish))
        changed = true;
    }
  }

  return changed;
}

} // namespace luci
