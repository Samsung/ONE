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

#include "luci/Pass/FuseAddToFullyConnectedBiasPass.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Profile/CircleNodeOrigin.h>

#include "helpers/NodeFiller.h"

#define CHECK_OR_FALSE(condition) \
  if (not(condition))             \
    return false;

namespace
{

/**
 *  Fuse Add to following FullyConnected bias if possible
 *
 *  BEFORE
 *                |
 *           [CircleAdd] [CircleConst] [CircleConst]
 *                |       |              |
 *      [CircleFullyConnected] ----------+
 *                |
 *
 *  AFTER
 *                |
 *                |        [CircleConst] [CircleConst] [CircleConst]
 *                |                   |       |         |
 *                |   [CircleConst] [CircleFullyConnected]   [CircleAdd]
 *                |       |           |
 *       [CircleFullyConnected] ------+
 *                |
 *
 */
bool fuse_add_to_fc_bias(luci::CircleFullyConnected *fc)
{
  CHECK_OR_FALSE(fc);

  // check input is Add
  auto add = dynamic_cast<luci::CircleAdd *>(fc->input());
  CHECK_OR_FALSE(add);
  // conditions of Add, FC: to expect constant folding, support only F32
  CHECK_OR_FALSE(add->dtype() == loco::DataType::FLOAT32);
  CHECK_OR_FALSE(add->fusedActivationFunction() == luci::FusedActFunc::NONE);
  CHECK_OR_FALSE(fc->dtype() == loco::DataType::FLOAT32);
  // support weight with constant
  auto weights = dynamic_cast<luci::CircleConst *>(fc->weights());
  CHECK_OR_FALSE(weights);
  // bias can be constant or outputexclude
  auto bias = dynamic_cast<luci::CircleNode *>(fc->bias());
  CHECK_OR_FALSE(bias);

  // Check addition of Add is constant
  luci::CircleNode *add_input = nullptr;
  luci::CircleConst *add_shift = nullptr;
  CHECK_OR_FALSE(luci::fill(&add_input, &add_shift).with_commutative_args_of(add));
  // support only 1D constant
  CHECK_OR_FALSE(add_shift->rank() == 1);

  auto graph = fc->graph();

  auto fc_bias = graph->nodes()->create<luci::CircleFullyConnected>();
  fc_bias->input(add_shift);
  fc_bias->weights(weights);
  fc_bias->bias(bias);
  fc_bias->keep_num_dims(true);
  fc_bias->fusedActivationFunction(luci::FusedActFunc::NONE);
  fc_bias->name(fc->name() + "_" + add->name() + "_bias");
  luci::add_origin(fc_bias,
                   luci::composite_origin(
                     {luci::get_origin(add), luci::get_origin(add_shift), luci::get_origin(bias)}));

  auto fc_new = graph->nodes()->create<luci::CircleFullyConnected>();
  fc_new->input(add_input);
  fc_new->weights(weights);
  fc_new->bias(fc_bias);
  fc_new->weights_format(fc->weights_format());
  fc_new->keep_num_dims(fc->keep_num_dims());
  fc_new->fusedActivationFunction(fc->fusedActivationFunction());
  fc_new->name(fc->name());
  luci::add_origin(fc_new, luci::get_origin(fc));

  replace(fc).with(fc_new);

  return true;
}

} // namespace

namespace luci
{

bool FuseAddToFullyConnectedBiasPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto fc = dynamic_cast<luci::CircleFullyConnected *>(node);
    if (not fc)
      continue;

    if (fuse_add_to_fc_bias(fc))
      changed = true;
  }

  return changed;
}

} // namespace luci
