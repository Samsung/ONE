/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/FuseMulWithRmsNormPass.h"

#include "helpers/NodeFiller.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Service/Nodes/CircleConst.h>
#include <luci/Profile/CircleNodeOrigin.h>

namespace
{

#define RETURN_FALSE_UNLESS(cond) \
  if (not(cond))                  \
    return false;

/**
 *  Fuse Mul to RmsNorm if the RmsNorm's weight value is 1.0
 *
 *  BEFORE
 *                |
 *         [CircleRmsNorm] (gamma=1.0)
 *                |
 *           [CircleMul]
 *                |
 *
 *  AFTER
 *                |
 *         [CircleRmsNorm]
 *                |
 *
 */
bool fuse_mul_with_rmsnorm(luci::CircleMul *mul)
{
  RETURN_FALSE_UNLESS(mul);
  RETURN_FALSE_UNLESS(mul->dtype() == loco::DataType::FLOAT32);

  luci::CircleRmsNorm *rmsnorm = nullptr;
  luci::CircleConst *weight = nullptr;
  RETURN_FALSE_UNLESS(luci::fill(&rmsnorm, &weight).with_commutative_args_of(mul));

  RETURN_FALSE_UNLESS(loco::succs(rmsnorm).size() == 1);
  RETURN_FALSE_UNLESS(rmsnorm->dtype() == loco::DataType::FLOAT32);

  auto norm_gamma = dynamic_cast<luci::CircleConst *>(rmsnorm->gamma());
  RETURN_FALSE_UNLESS(norm_gamma);
  RETURN_FALSE_UNLESS(norm_gamma->size<loco::DataType::FLOAT32>() == 1);
  RETURN_FALSE_UNLESS(norm_gamma->at<loco::DataType::FLOAT32>(0) == 1.0f);

  RETURN_FALSE_UNLESS(weight->rank() == 1)
  RETURN_FALSE_UNLESS(weight->dim(0) == rmsnorm->dim(rmsnorm->rank() - 1));

  auto fused_weight = luci::clone(weight);
  rmsnorm->gamma(fused_weight);

  luci::add_origin(rmsnorm, luci::get_origin(mul));

  replace(mul).with(rmsnorm);

  return true;
}

} // namespace

namespace luci
{

bool FuseMulWithRmsNormPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto mul = dynamic_cast<luci::CircleMul *>(node))
    {
      if (fuse_mul_with_rmsnorm(mul))
        changed = true;
    }
  }

  return changed;
}

} // namespace luci
