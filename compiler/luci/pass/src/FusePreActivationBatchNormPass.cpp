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

#include "luci/Pass/FusePreActivationBatchNormPass.h"
#include "FusePreActivationBatchNormPassInternal.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Log.h>

namespace
{

// Check if all elements are non-negative
bool is_non_negative(const luci::CircleConst *node)
{
  assert(node->dtype() == loco::DataType::FLOAT32);

  uint32_t size = node->size<loco::DataType::FLOAT32>();
  for (uint32_t i = 0; i < size; ++i)
  {
    if (node->at<loco::DataType::FLOAT32>(i) < 0)
      return false;
  }
  return true;
}

// Check if mul is batchnorm mul
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
  if (!(channel_dim == mul->dim(mul->rank() - 1)))
    return false;

  pred_node = pred;
  gamma = constant;
  return true;
}

// Check if add is batchnorm add
bool is_batchnorm_add(const luci::CircleAdd *add, luci::CircleMul *&mul, luci::CircleConst *&beta)
{
  auto x = loco::must_cast<luci::CircleNode *>(add->x());
  auto y = loco::must_cast<luci::CircleNode *>(add->y());

  luci::CircleMul *pred = nullptr;
  luci::CircleConst *constant = nullptr;

  if (add->fusedActivationFunction() != luci::FusedActFunc::RELU)
    return false;

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

} // namespace

namespace luci
{

/**
 *  Swap MUL/ADD if they are from batch normalization
 *
 *  BEFORE
 *           [Mul]  gamma
 *             |
 *        [Add + Relu]  beta
 *
 *  AFTER
 *           [Add]  beta/gamma
 *             |
 *           [Mul]  gamma
 *             |
 *           [Relu]
 */
bool swap_mul_add(luci::CircleAdd *add, std::vector<luci::CircleMul *> &mul_list,
                  std::vector<luci::CircleAdd *> &add_list)
{
  luci::CircleNode *pred_node = nullptr;
  luci::CircleMul *mul = nullptr;
  luci::CircleConst *beta = nullptr;
  luci::CircleConst *gamma = nullptr;

  if (!is_batchnorm_add(add, mul, beta))
    return false;

  if (loco::succs(mul).size() != 1)
    return false;

  if (!is_batchnorm_mul(mul, pred_node, gamma))
    return false;

  if (beta->dtype() != loco::DataType::FLOAT32 || gamma->dtype() != loco::DataType::FLOAT32)
    throw std::runtime_error("FusePreActivationBatchNormPass only supports Float32 model");

  if (!is_non_negative(gamma))
    return false;

  // Insert Relu at the bottom
  auto relu = add->graph()->nodes()->create<luci::CircleRelu>();
  relu->features(mul);
  loco::replace(add).with(relu);

  // Replace beta <- beta / gamma
  if (add->x() == beta)
  {
    add->y(pred_node);
  }
  else
  {
    add->x(pred_node);
  }
  add->fusedActivationFunction(luci::FusedActFunc::NONE);
  uint32_t size = beta->size<loco::DataType::FLOAT32>();
  for (uint32_t i = 0; i < size; ++i)
  {
    auto b = beta->at<loco::DataType::FLOAT32>(i);
    auto g = gamma->at<loco::DataType::FLOAT32>(i);
    if (b == g)
    {
      beta->at<loco::DataType::FLOAT32>(i) = 1;
    }
    else
    {
      // If g is 0, we use a small value (empirically determined)
      if (g == 0)
        g = 1e-10;
      beta->at<loco::DataType::FLOAT32>(i) = b / g;
    }
  }

  if (mul->x() == gamma)
  {
    mul->y(add);
  }
  else
  {
    mul->x(add);
  }

  mul_list.push_back(mul);
  add_list.push_back(add);

  return true;
}

bool FusePreActivationBatchNormPass::run(loco::Graph *g)
{
  LOGGER(l);
  bool changed = false;

  // Step 1. Swap MUL <-> ADD
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto add = dynamic_cast<luci::CircleAdd *>(node);
    if (add == nullptr)
      continue;

    if (swap_mul_add(add, _mul_list, _add_list))
      changed = true;
  }

  INFO(l) << "[FusePreActivationBatchNorm] Target pre-activations: " << _mul_list.size()
          << std::endl;

  // Valid pattern was not detected. Fast exit.
  if (!changed)
    return false;

  // Step 2. Fuse MUL with the next CONV

  // Step 3. Fuse ADD with the preceding CONV and insert SUB

  // Step 4. Fuse SUB to CONV (SUB -> ADD <- CONV pattern)

  return changed;
}

} // namespace luci
