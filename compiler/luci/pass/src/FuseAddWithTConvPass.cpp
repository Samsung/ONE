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

#include "luci/Pass/FuseAddWithTConvPass.h"

#include "helpers/NodeFiller.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Profile/CircleNodeOrigin.h>

namespace
{

#define RETURN_FALSE_UNLESS(cond) \
  if (not(cond))                  \
    return false;

/**
 *  Fuse Add to TransposeConv if possible
 *
 *  BEFORE
 *                     |
 *   [CircleConst]  [CircleTransposeConv]
 *               \     |
 *             [CircleAdd]
 *                  |
 *
 *  AFTER
 *                  |
 *   [CircleConst]  |
 *             \    |
 *         [CircleTransposeConv]   [CircleAdd]
 *                  |
 *          ([CircleRelu/Relu6])
 *                  |
 *
 *  Note: CircleRelu/Relu6 is inserted if Add activation is ReLU6
 */
bool fuse_add_with_tconv(luci::CircleAdd *add)
{
  // Allow Add node only with FLOAT32 data type.
  RETURN_FALSE_UNLESS(add->dtype() == loco::DataType::FLOAT32);
  // Allow Add node only with specific activations.
  RETURN_FALSE_UNLESS(add->fusedActivationFunction() == luci::FusedActFunc::NONE ||
                      add->fusedActivationFunction() == luci::FusedActFunc::RELU6 ||
                      add->fusedActivationFunction() == luci::FusedActFunc::RELU);
  // Find the pattern of Add(TransposeConv, CircleConst):
  luci::CircleTransposeConv *tconv = nullptr;
  luci::CircleConst *addition = nullptr;
  RETURN_FALSE_UNLESS(luci::fill(&tconv, &addition).with_commutative_args_of(add));

  RETURN_FALSE_UNLESS(loco::succs(tconv).size() == 1);

  // Skip if tconv has fused activation.
  RETURN_FALSE_UNLESS(tconv->fusedActivationFunction() == luci::FusedActFunc::NONE);
  // Check whether tconv has bias or not. This optimization works only if it doesn't.
  auto bias = dynamic_cast<luci::CircleOutputExclude *>(tconv->bias());
  RETURN_FALSE_UNLESS(bias);
  // Get weights of tconv:
  auto filter = dynamic_cast<luci::CircleConst *>(tconv->filter());
  RETURN_FALSE_UNLESS(filter);
  RETURN_FALSE_UNLESS(filter->dtype() == loco::DataType::FLOAT32);

  // addition dim(0) == tconv filter channel dim
  RETURN_FALSE_UNLESS(addition->rank() == 1);

  auto addition_dim = addition->dim(0).value();
  auto filter_channel_dim = filter->dim(0).value();
  RETURN_FALSE_UNLESS(filter_channel_dim == addition_dim);

  // Fuse addition with transposed conv:
  tconv->bias(addition);

  if (add->fusedActivationFunction() == luci::FusedActFunc::RELU6)
  {
    auto name = addition->name();
    assert(name.length() > 0);
    // Separate relu op from add op:
    auto relu = add->graph()->nodes()->create<luci::CircleRelu6>();
    relu->features(tconv);
    relu->name(name + "/Relu6");
    luci::add_origin(relu, luci::get_origin(add));

    // Remove add node.
    replace(add).with(relu);
  }
  else if (add->fusedActivationFunction() == luci::FusedActFunc::RELU)
  {
    auto name = addition->name();
    assert(name.length() > 0);
    // Separate relu op from add op:
    auto relu = add->graph()->nodes()->create<luci::CircleRelu>();
    relu->features(tconv);
    relu->name(name + "/Relu");
    luci::add_origin(relu, luci::get_origin(add));

    // Remove add node.
    replace(add).with(relu);
  }
  else
  {
    // Remove add node.
    replace(add).with(tconv);
  }

  // Set new origin.
  luci::add_origin(tconv, luci::get_origin(add));

  return true;
}

} // namespace

namespace luci
{

bool FuseAddWithTConvPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto add = dynamic_cast<luci::CircleAdd *>(node))
      if (fuse_add_with_tconv(add))
        changed = true;
  }

  return changed;
}

} // namespace luci
