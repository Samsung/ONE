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

const luci::CircleConv2D *get_forward_conv2d(const luci::CircleNode *node, uint32_t channel_size)
{
  auto opcode = node->opcode();
  if (opcode == luci::CircleOpcode::CONV_2D)
  {
    auto conv = loco::must_cast<const luci::CircleConv2D *>(node);
    auto filter = dynamic_cast<luci::CircleConst *>(conv->filter());
    if (filter == nullptr)
      return nullptr;

    if (filter->rank() != 4)
      return nullptr;

    if (filter->dim(3).value() != channel_size)
      return nullptr;

    if (loco::succs(filter).size() != 1)
      return nullptr;

    return conv;
  }
  // MUL can be fused with CONV across MEAN
  // i.e., MUL-MEAN-CONV -> MEAN-CONV
  // This is for handling the last part of ResNetV2
  else if (opcode == luci::CircleOpcode::MEAN)
  {
    auto mean = loco::must_cast<const luci::CircleMean *>(node);
    auto axis = mean->reduction_indices();
    auto axis_const = dynamic_cast<luci::CircleConst *>(axis);
    if (not axis_const)
      return nullptr;

    assert(axis_const->dtype() == loco::DataType::S32);
    auto axis_size = axis_const->size<loco::DataType::S32>();
    for (uint32_t i = 0; i < axis_size; ++i)
    {
      // Reduction axis must not be the channel index
      // Assumption: Layout is channel-last
      if (axis_const->at<loco::DataType::S32>(i) == static_cast<int32_t>(node->rank() - 1))
        return nullptr;
    }

    auto succ = loco::succs(node);
    if (succ.size() != 1)
      return nullptr;

    auto succ_node = loco::must_cast<luci::CircleNode *>(*succ.begin());

    return get_forward_conv2d(succ_node, channel_size);
  }
  else
  {
    return nullptr;
  }
}

void update_conv_weights_with_gamma(const luci::CircleConv2D *conv, const luci::CircleConst *gamma)
{
  assert(conv != nullptr);
  assert(gamma != nullptr);
  auto filter = loco::must_cast<luci::CircleConst *>(conv->filter());

  uint32_t filter_out_dim = filter->dim(0).value();
  uint32_t filter_height_dim = filter->dim(1).value();
  uint32_t filter_width_dim = filter->dim(2).value();
  uint32_t filter_in_dim = filter->dim(3).value();
  for (uint32_t o = 0; o < filter_out_dim; o++)
  {
    for (uint32_t h = 0; h < filter_height_dim; h++)
    {
      for (uint32_t w = 0; w < filter_width_dim; w++)
      {
        for (uint32_t i = 0; i < filter_in_dim; i++)
        {
          uint32_t offset = o * filter_height_dim * filter_width_dim * filter_in_dim +
                            h * filter_width_dim * filter_in_dim + w * filter_in_dim + i;
          filter->at<loco::DataType::FLOAT32>(offset) *= gamma->at<loco::DataType::FLOAT32>(i);
        }
      }
    }
  }
}

// Find CONV_2D that can be fused with ADD
luci::CircleConv2D *get_backward_conv2d(luci::CircleNode *node, uint32_t channel_size)
{
  // Stop searching when meeting a node used by multiple nodes
  if (loco::succs(node).size() != 1)
    return nullptr;

  auto opcode = node->opcode();
  if (opcode == luci::CircleOpcode::CONV_2D)
  {
    auto conv = loco::must_cast<luci::CircleConv2D *>(node);
    auto filter = dynamic_cast<luci::CircleConst *>(conv->filter());

    if (filter == nullptr)
      return nullptr;

    if (filter->rank() != 4)
      return nullptr;

    if (filter->dim(0).value() != channel_size)
      return nullptr;

    if (loco::succs(filter).size() != 1)
      return nullptr;

    return conv;
  }
  else if (opcode == luci::CircleOpcode::MAX_POOL_2D || opcode == luci::CircleOpcode::PAD ||
           opcode == luci::CircleOpcode::ADD)
  {
    auto preds = loco::preds(node);
    for (auto pred : preds)
    {
      auto pred_conv = get_backward_conv2d(loco::must_cast<luci::CircleNode *>(pred), channel_size);
      if (pred_conv != nullptr)
        return pred_conv;
    }
    return nullptr;
  }
  else
  {
    return nullptr;
  }
}

bool update_conv_bias_with_beta(luci::CircleConv2D *conv, const luci::CircleConst *beta,
                                bool add_beta)
{
  assert(beta->rank() == 1);
  auto size = beta->dim(0).value();
  auto bias = dynamic_cast<luci::CircleConst *>(conv->bias());

  if (bias == nullptr)
  {
    bias = conv->graph()->nodes()->create<luci::CircleConst>();
    bias->dtype(loco::DataType::FLOAT32);
    bias->rank(1);
    bias->dim(0).set(size);
    bias->size<loco::DataType::FLOAT32>(size);
    conv->bias(bias);
  }
  else
  {
    if (bias->rank() != 1)
      return false;

    if (loco::succs(bias).size() != 1)
      return false;

    if (size != bias->dim(0).value())
      return false;
  }

  for (uint32_t i = 0; i < size; i++)
  {
    if (add_beta)
      bias->at<loco::DataType::FLOAT32>(i) += beta->at<loco::DataType::FLOAT32>(i);
    else
      bias->at<loco::DataType::FLOAT32>(i) -= beta->at<loco::DataType::FLOAT32>(i);
  }
  return true;
}

luci::CircleSub *insert_sub(luci::CircleNode *pred, luci::CircleConst *beta)
{
  auto sub = pred->graph()->nodes()->create<luci::CircleSub>();
  sub->dtype(loco::DataType::FLOAT32);
  sub->rank(pred->rank());
  for (uint32_t i = 0; i < sub->rank(); i++)
  {
    sub->dim(i).set(pred->dim(i).value()); // TODO : value() fix needed
  }
  sub->fusedActivationFunction(luci::FusedActFunc::NONE);

  loco::replace(pred).with(sub);

  sub->x(pred);
  sub->y(beta);

  return sub;
}

luci::CircleAdd *get_forward_add(luci::CircleNode *node)
{
  auto opcode = node->opcode();
  if (opcode == luci::CircleOpcode::ADD)
  {
    auto add = loco::must_cast<luci::CircleAdd *>(node);
    return add;
  }
  else if (opcode == luci::CircleOpcode::MAX_POOL_2D)
  {
    auto succ = loco::succs(node);
    if (succ.size() != 1)
      return nullptr;

    auto succ_node = loco::must_cast<luci::CircleNode *>(*succ.begin());
    return get_forward_add(succ_node);
  }

  return nullptr;
}

} // namespace

namespace luci
{

/**
 *  Fuse SUB with CONV
 *
 *  BEFORE
 *
 *     beta [Sub]
 *            |
 *      [Passable Op]   [Conv] bias
 *                 \   /
 *                 [Add]
 *
 *  AFTER
 *
 *      [Passable Op]   [Conv] bias - beta
 *                 \   /
 *                 [Add]
 */
bool fuse_sub_with_conv(luci::CircleSub *sub)
{
  luci::CircleAdd *add = nullptr;
  luci::CircleConv2D *conv = nullptr;
  auto succs = loco::succs(sub);
  if (succs.size() != 1)
    return false;

  add = get_forward_add(loco::must_cast<luci::CircleNode *>(*succs.begin()));
  if (add == nullptr)
    return false;

  conv = dynamic_cast<luci::CircleConv2D *>(add->x());
  if (conv == nullptr)
    conv = dynamic_cast<luci::CircleConv2D *>(add->y());

  if (conv == nullptr)
    return false;

  auto beta = loco::must_cast<luci::CircleConst *>(sub->y());
  assert(beta != nullptr);
  if (!update_conv_bias_with_beta(conv, beta, false))
    return false;

  auto pred = sub->x();
  loco::replace(sub).with(pred);

  sub->drop();

  return true;
}

/**
 *  Fuse ADD with the preceding CONV
 *
 *  BEFORE
 *
 *        [Conv]  bias
 *           |
 *     [Passable Op] (None, Max pool, Pad, etc)
 *           |
 *         [Add]  beta
 *
 *  AFTER
 *
 *        [Conv]  bias + beta
 *           |
 *     [Passable Op]
 *
 *  A special case where SUB is newly inserted
 *
 *  BEFORE
 *
 *              [Conv]  bias
 *           \   /
 *           [Add]
 *           /   \
 *              [Add]  beta
 *
 *  AFTER
 *
 *              [Conv]  bias + beta
 *           \   /
 *           [Add]
 *           /   \
 *   beta [Sub]
 */
bool fuse_add_with_conv(luci::CircleAdd *add, std::vector<luci::CircleSub *> &sub_list)
{
  auto x = dynamic_cast<luci::CircleConst *>(add->x());
  auto y = dynamic_cast<luci::CircleConst *>(add->y());

  luci::CircleNode *pred = nullptr;
  luci::CircleConst *beta = nullptr;

  if (x != nullptr && y == nullptr)
  {
    pred = loco::must_cast<luci::CircleNode *>(add->y());
    beta = x;
  }
  else if (x == nullptr && y != nullptr)
  {
    pred = loco::must_cast<luci::CircleNode *>(add->x());
    beta = y;
  }
  else
  {
    return false;
  }

  assert(beta->rank() == 1);

  auto channel_size = beta->dim(0).value();
  auto conv = get_backward_conv2d(pred, channel_size);

  if (conv != nullptr)
  {
    if (!update_conv_bias_with_beta(conv, beta, true))
      return false;

    loco::replace(add).with(pred);
    add->drop();

    return true;
  }
  // A special case shown at the residual blocks of ResNetV2
  // TODO: Handle this with get_backward_conv2d
  else if (pred->opcode() == luci::CircleOpcode::ADD)
  {
    auto pred_add = loco::must_cast<luci::CircleAdd *>(pred);
    conv = get_backward_conv2d(loco::must_cast<luci::CircleNode *>(pred_add->y()), channel_size);
    if (conv == nullptr)
      conv = get_backward_conv2d(loco::must_cast<luci::CircleNode *>(pred_add->x()), channel_size);

    if (conv == nullptr)
      return false;

    if (!update_conv_bias_with_beta(conv, beta, true))
      return false;

    auto relu = *loco::succs(add).begin();
    auto relu_node = loco::must_cast<luci::CircleRelu *>(relu);
    assert(relu_node != nullptr);

    loco::replace(add).with(pred);

    add->drop();

    sub_list.push_back(insert_sub(pred, beta));

    relu_node->features(pred);

    return true;
  }

  return false;
}

/**
 *  Fuse MUL with the next CONV
 *
 *  BEFORE
 *
 *           [Mul]  gamma
 *             |
 *           [Relu]
 *            /  \
 *     W1 [Conv]  [Conv] W2
 *
 *  AFTER
 *
 *                [Relu]
 *                 /  \
 *   gamma X W1 [Conv]  [Conv] gamma X W2
 */
bool fuse_mul_with_conv(luci::CircleMul *mul)
{
  luci::CircleNode *pred_node = nullptr;
  luci::CircleConst *gamma = nullptr;

  if (!is_batchnorm_mul(mul, pred_node, gamma))
    return false;

  auto mul_succ = loco::succs(mul);
  assert(mul_succ.size() == 1);

  auto relu = loco::must_cast<luci::CircleRelu *>(*mul_succ.begin());

  auto channel_size = gamma->dim(0).value();

  bool fusable = true;
  auto relu_succ = loco::succs(relu);
  for (auto s : relu_succ)
  {
    auto conv = get_forward_conv2d(loco::must_cast<luci::CircleNode *>(s), channel_size);
    if (conv == nullptr)
      fusable = false;
  }

  if (fusable)
  {
    for (auto s : relu_succ)
    {
      // Find the next CONV
      auto conv = get_forward_conv2d(loco::must_cast<luci::CircleNode *>(s), channel_size);

      // Update CONV weights
      update_conv_weights_with_gamma(conv, gamma);
    }

    loco::replace(mul).with(pred_node);
    relu->features(pred_node);

    mul->drop();

    return true;
  }

  return false;
}

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
  for (auto const &mul : _mul_list)
  {
    if (fuse_mul_with_conv(mul))
      INFO(l) << "[FusePreActivationBatchNorm] Fused MUL: " << mul->name() << std::endl;
  }

  // Step 3. Fuse ADD with the preceding CONV and insert SUB
  for (auto const &add : _add_list)
  {
    if (fuse_add_with_conv(add, _sub_list))
      INFO(l) << "[FusePreActivationBatchNorm] Fused ADD: " << add->name() << std::endl;
  }

  INFO(l) << "[FusePreActivationBatchNorm] " << _sub_list.size() << " SUB were added." << std::endl;

  // Step 4. Fuse SUB to CONV (SUB -> ADD <- CONV pattern)
  for (auto const &sub : _sub_list)
  {
    if (fuse_sub_with_conv(sub))
      INFO(l) << "[FusePreActivationBatchNorm] Fused SUB: " << sub->name() << std::endl;
  }

  return changed;
}

} // namespace luci
