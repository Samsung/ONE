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

#include "luci/Pass/ReplaceMulAddWithDepthwiseConvPass.h"

#include <luci/IR/CircleNodes.h>

namespace
{

luci::CircleConst *create_weights_from_gamma(luci::CircleConst *gamma)
{
  assert(gamma->rank() == 1);
  auto channel_size = gamma->dim(0).value();

  // Channel-wise MUL is the same as DEPTHWISE_CONV2D with filter shape (1,1,1,channel_size)
  auto weights = gamma->graph()->nodes()->create<luci::CircleConst>();
  weights->dtype(loco::DataType::FLOAT32);
  weights->rank(4);
  weights->dim(0).set(1);
  weights->dim(1).set(1);
  weights->dim(2).set(1);
  weights->dim(3).set(channel_size);
  weights->shape_status(luci::ShapeStatus::VALID);
  weights->size<loco::DataType::FLOAT32>(channel_size);
  for (uint32_t i = 0; i < channel_size; i++)
  {
    weights->at<loco::DataType::FLOAT32>(i) = gamma->at<loco::DataType::FLOAT32>(i);
  }

  return weights;
}

luci::CircleConst *create_bias_from_beta(luci::CircleConst *beta)
{
  assert(beta->rank() == 1);
  auto channel_size = beta->dim(0).value();

  // Channel-wise ADD is the same as bias (shape = (channel_size)) of DEPTHWISE_CONV2D
  auto bias = beta->graph()->nodes()->create<luci::CircleConst>();
  bias->dtype(loco::DataType::FLOAT32);
  bias->rank(1);
  bias->dim(0).set(channel_size);
  bias->size<loco::DataType::FLOAT32>(channel_size);
  bias->shape_status(luci::ShapeStatus::VALID);
  for (uint32_t i = 0; i < channel_size; i++)
  {
    bias->at<loco::DataType::FLOAT32>(i) = beta->at<loco::DataType::FLOAT32>(i);
  }

  return bias;
}

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
  if (!(channel_dim == add->dim(add->rank() - 1))) // TODO Which value should be selected for unknown?
    return false;

  mul = pred;
  beta = constant;
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
  if (!(channel_dim == mul->dim(mul->rank() - 1))) // TODO Which value should be selected for unknown?
    return false;

  pred_node = pred;
  gamma = constant;
  return true;
}

/**
 *  Replace channel-wise Mul/Add with DepthwiseConv2D
 *
 *  BEFORE
 *
 *             [Node] [gamma]
 *                |  /
 *              [Mul]  [beta]
 *                |   /
 *               [Add]
 *
 *  AFTER
 *
 *              [Node]  [weights]  [bias]
 *                  \      /       /
 *                [DepthwiseConv2D]
 */
bool replace_mul_add_with_dwconv(luci::CircleAdd *add)
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

  if (pred_node->rank() != 4)
    return false;

  if (pred_node->dtype() != loco::DataType::FLOAT32 || beta->dtype() != loco::DataType::FLOAT32 ||
      gamma->dtype() != loco::DataType::FLOAT32)
    return false;

  auto weights = create_weights_from_gamma(gamma);
  auto bias = create_bias_from_beta(beta);

  auto dwconv = add->graph()->nodes()->create<luci::CircleDepthwiseConv2D>();
  dwconv->input(pred_node);
  dwconv->filter(weights);
  dwconv->bias(bias);
  dwconv->padding(luci::Padding::SAME);
  dwconv->stride()->w(1);
  dwconv->stride()->h(1);
  dwconv->depthMultiplier(1);
  dwconv->dilation()->w(1);
  dwconv->dilation()->h(1);
  dwconv->fusedActivationFunction(add->fusedActivationFunction());

  loco::replace(add).with(dwconv);
  return true;
}

} // namespace

namespace luci
{

bool ReplaceMulAddWithDepthwiseConvPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto add = dynamic_cast<luci::CircleAdd *>(node);
    if (not add)
      continue;

    if (replace_mul_add_with_dwconv(add))
    {
      changed = true;
      break;
    }
  }

  return changed;
}

} // namespace luci
