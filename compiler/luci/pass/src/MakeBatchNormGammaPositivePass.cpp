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

#include "luci/Pass/MakeBatchNormGammaPositivePass.h"

#include "BatchNormPatternFinder.h"

#include <luci/IR/CircleNodes.h>

namespace
{

// Update negative gamma to positive (1e-10)
bool negative_gamma_to_positive(luci::CircleConst *gamma)
{
  assert(gamma->dtype() == loco::DataType::FLOAT32);

  bool changed = false;
  uint32_t size = gamma->size<loco::DataType::FLOAT32>();
  for (uint32_t i = 0; i < size; ++i)
  {
    if (gamma->at<loco::DataType::FLOAT32>(i) < 0)
    {
      gamma->at<loco::DataType::FLOAT32>(i) = 1e-10;
      changed = true;
    }
  }
  return changed;
}

bool make_positive_gamma(luci::CircleAdd *add)
{
  luci::CircleMul *mul = nullptr;
  luci::CircleConst *beta = nullptr;
  luci::CircleConst *gamma = nullptr;
  luci::CircleNode *pred = nullptr;

  if (!is_batchnorm_add(add, mul, beta))
    return false;

  if (loco::succs(mul).size() != 1)
    return false;

  if (!is_batchnorm_mul(mul, pred, gamma))
    return false;
  assert(pred == add);
  // Only support Relu
  if (add->fusedActivationFunction() != luci::FusedActFunc::RELU)
    return false;

  return negative_gamma_to_positive(gamma);
}

} // namespace

namespace luci
{

/**
 * Make gamma value of Mul-Add(as BatchNorm) to positive
 *
 *  PATTERN:
 *          |
 *    [CircleNode] [CircleConst](as gamma)
 *              |   |
 *           [CircleMul] [CircleConst]
 *                   |    |
 *               [CircleAdd]
 *                     |
 */
bool MakeBatchNormGammaPositivePass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto add = dynamic_cast<luci::CircleAdd *>(node);
    if (add == nullptr)
      continue;

    if (make_positive_gamma(add))
      changed = true;
  }
  return changed;
}

} // namespace luci
