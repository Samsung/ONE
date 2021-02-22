/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "BatchNormPatternFinder.h"

#include <luci/IR/CircleNodes.h>

namespace luci
{

bool is_batchnorm_add(const luci::CircleAdd *add, luci::CircleMul *&mul, luci::CircleConst *&beta)
{
  auto x = loco::must_cast<luci::CircleNode *>(add->x());
  auto y = loco::must_cast<luci::CircleNode *>(add->y());

  luci::CircleMul *pred = nullptr;
  luci::CircleConst *constant = nullptr;

  if (x->opcode() == luci::CircleOpcode::CIRCLECONST && y->opcode() == luci::CircleOpcode::MUL)
  {
    pred = loco::must_cast<luci::CircleMul *>(y);
    constant = loco::must_cast<luci::CircleConst *>(x);
  }
  else if (x->opcode() == luci::CircleOpcode::MUL && y->opcode() == luci::CircleOpcode::CIRCLECONST)
  {
    pred = loco::must_cast<luci::CircleMul *>(x);
    constant = loco::must_cast<luci::CircleConst *>(y);
  }
  else
  {
    return false;
  }

  if (constant->rank() != 1)
    return false;

  auto channel_dim = constant->dim(0);
  // Assumption: Layout is channel-last
  if (!(channel_dim == add->dim(add->rank() - 1)))
    return false;

  mul = pred;
  beta = constant;
  return true;
}

bool is_batchnorm_add(const luci::CircleAdd *add)
{
  // for dummy mul and beta
  luci::CircleMul *mul = nullptr;
  luci::CircleConst *beta = nullptr;

  return is_batchnorm_add(add, mul, beta);
}

bool is_batchnorm_mul(const luci::CircleMul *mul, luci::CircleNode *&pred_node,
                      luci::CircleConst *&gamma)
{
  auto x = dynamic_cast<luci::CircleConst *>(mul->x());
  auto y = dynamic_cast<luci::CircleConst *>(mul->y());

  luci::CircleNode *pred = nullptr;
  luci::CircleConst *constant = nullptr;

  if (x != nullptr && y == nullptr)
  {
    pred = loco::must_cast<luci::CircleNode *>(mul->y());
    constant = x;
  }
  else if (x == nullptr && y != nullptr)
  {
    pred = loco::must_cast<luci::CircleNode *>(mul->x());
    constant = y;
  }
  else
  {
    return false;
  }

  if (constant->rank() != 1)
    return false;

  auto channel_dim = constant->dim(0);
  // Assumption: Layout is channel-last
  if (!(channel_dim == mul->dim(mul->rank() - 1)))
    return false;

  pred_node = pred;
  gamma = constant;
  return true;
}

} // namespace luci
