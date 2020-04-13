/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "FuseSquaredDifferencePass.h"

#include "Check.h"

#include "Dialect/IR/TFLNodes.h"

namespace
{

/**
 * @return  Casted TFLMul for fusable candidate, nullptr otherwise
 *
 * This helper checkes fusability with following conditions:
 * - TFLMul has no activation
 * - TFLMul's first and second arguments are equal and TFLSub
 */
locoex::TFLMul *as_candidate(loco::Node *node)
{
  auto mul = dynamic_cast<locoex::TFLMul *>(node);
  if (not mul)
    return nullptr;

  // Cannot fuse mul with activation function
  if (mul->fusedActivationFunction() != locoex::FusedActFunc::NONE)
    return nullptr;

  if (mul->x() != mul->y())
    return nullptr;

  if (not dynamic_cast<locoex::TFLSub *>(mul->x()))
    return nullptr;

  return mul;
}

void fuse_squared_difference(locoex::TFLMul *mul)
{
  auto sub = dynamic_cast<locoex::TFLSub *>(mul->x());
  EXO_ASSERT(sub, "sub should be valid at this point");

  // TFLSquaredDifference to replace
  auto sq_diff = mul->graph()->nodes()->create<locoex::TFLSquaredDifference>();
  sq_diff->x(sub->x());
  sq_diff->y(sub->y());

  // replace
  loco::replace(mul).with(sq_diff);
}

} // namespace

namespace exo
{

bool FuseSquaredDifferencePass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto mul = as_candidate(node))
    {
      fuse_squared_difference(mul);
      changed = true;
    }
  }

  return changed;
}

} // namespace exo
