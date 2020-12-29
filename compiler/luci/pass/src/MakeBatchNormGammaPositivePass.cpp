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

// Check if add is batchnorm add
bool is_batchnorm_add(const luci::CircleAdd *add)
{
  auto x = dynamic_cast<luci::CircleConst *>(add->x());
  auto y = dynamic_cast<luci::CircleConst *>(add->y());

  luci::CircleConst *constant = nullptr;

  if (x != nullptr && y == nullptr)
    constant = x;
  else if (x == nullptr && y != nullptr)
    constant = y;
  else
    return false;

  if (constant->rank() != 1)
    return false;

  // Only support Relu
  if (add->fusedActivationFunction() != luci::FusedActFunc::RELU)
    return false;

  auto channel_dim = constant->dim(0);
  if (!(channel_dim == add->dim(add->rank() - 1))) // TODO Which value should be selected for unknown?
    return false;

  return true;
}

// Check if mul is batchnorm mul
bool is_batchnorm_mul(const luci::CircleMul *mul, luci::CircleConst *&gamma)
{
  auto x = dynamic_cast<luci::CircleConst *>(mul->x());
  auto y = dynamic_cast<luci::CircleConst *>(mul->y());

  luci::CircleConst *constant = nullptr;

  if (x != nullptr && y == nullptr)
    constant = x;
  else if (x == nullptr && y != nullptr)
    constant = y;
  else
    return false;

  if (constant->rank() != 1)
    return false;

  auto channel_dim = constant->dim(0);
  if (!(channel_dim == mul->dim(mul->rank() - 1))) // TODO Which value should be selected for unknown?
    return false;

  // Check successor is batchnorm add
  auto succs = loco::succs(mul);
  if (succs.size() != 1)
    return false;

  auto add = dynamic_cast<luci::CircleAdd *>(*succs.begin());
  if (add == nullptr)
    return false;

  if (!is_batchnorm_add(add))
    return false;

  gamma = constant;
  return true;
}

} // namespace

namespace luci
{

bool MakeBatchNormGammaPositivePass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto mul = dynamic_cast<luci::CircleMul *>(node);
    if (mul == nullptr)
      continue;

    luci::CircleConst *gamma;
    if (is_batchnorm_mul(mul, gamma))
      changed = negative_gamma_to_positive(gamma);
  }
  return changed;
}

} // namespace luci
