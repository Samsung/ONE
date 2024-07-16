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

#include "luci/Pass/FuseMulToFullyConnectedWeightsPass.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Profile/CircleNodeOrigin.h>

#include "helpers/NodeFiller.h"

#define CHECK_OR_FALSE(condition) \
  if (not(condition))             \
    return false;

namespace
{

/**
 *  Fuse Mul to following FullyConnected if possible
 *
 *  BEFORE
 *                |
 *           [CircleMul] [CircleConst] [CircleConst]
 *                |       |              |
 *      [CircleFullyConnected] ----------+
 *                |
 *
 *  AFTER
 *                |
 *                | [CircleConst] [CircleConst]
 *                |         |       |
 *                |        [CircleMul] [CircleConst]   [CircleMul]
 *                |          |              |
 *       [CircleFullyConnected] ------------+
 *                |
 *
 */
bool fuse_fc_with_mul(luci::CircleFullyConnected *fc)
{
  CHECK_OR_FALSE(fc);

  // check input is Mul
  auto mul = dynamic_cast<luci::CircleMul *>(fc->input());
  CHECK_OR_FALSE(mul);
  // conditions of Mul, FC: to expect constant folding, support only F32
  CHECK_OR_FALSE(mul->dtype() == loco::DataType::FLOAT32);
  CHECK_OR_FALSE(mul->fusedActivationFunction() == luci::FusedActFunc::NONE);
  CHECK_OR_FALSE(fc->dtype() == loco::DataType::FLOAT32);
  // support weight with constant
  auto weights = dynamic_cast<luci::CircleConst *>(fc->weights());
  CHECK_OR_FALSE(weights);

  // Check multiplication of Mul is constant
  luci::CircleNode *mul_input = nullptr;
  luci::CircleConst *mul_scale = nullptr;
  CHECK_OR_FALSE(luci::fill(&mul_input, &mul_scale).with_commutative_args_of(mul));
  // support only 1D constant
  CHECK_OR_FALSE(mul_scale->rank() == 1);

  auto graph = fc->graph();

  auto fc_weights = graph->nodes()->create<luci::CircleMul>();
  fc_weights->x(weights);
  fc_weights->y(mul_scale);
  fc_weights->fusedActivationFunction(luci::FusedActFunc::NONE);
  fc_weights->name(mul->name() + "_" + fc->name() + "_weight");
  luci::add_origin(fc_weights,
                   luci::composite_origin({luci::get_origin(mul), luci::get_origin(weights),
                                           luci::get_origin(mul_scale)}));

  auto fc_new = graph->nodes()->create<luci::CircleFullyConnected>();
  fc_new->input(mul_input);
  fc_new->weights(fc_weights);
  fc_new->bias(fc->bias());
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

bool FuseMulToFullyConnectedWeightsPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto fc = dynamic_cast<luci::CircleFullyConnected *>(node);
    if (not fc)
      continue;

    if (fuse_fc_with_mul(fc))
      changed = true;
  }

  return changed;
}

} // namespace luci
