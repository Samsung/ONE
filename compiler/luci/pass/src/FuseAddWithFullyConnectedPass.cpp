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

#include "luci/Pass/FuseAddWithFullyConnectedPass.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Service/Nodes/CircleConst.h>
#include <luci/Profile/CircleNodeOrigin.h>

#include <cmath>

namespace
{
/**
 *  Fuse Add to FullyConnected if the added value is a channel(last dimension)-wise constant
 *
 *  BEFORE
 *                |
 *      [CircleFullyConnected]
 *                |
 *           [CircleAdd]
 *                |
 *
 *  AFTER
 *                |
 *       [CircleFullyConnected]   [CircleAdd] (dead)
 *                |
 *
 */
bool fuse_add_with_fc(luci::CircleFullyConnected *fc)
{
  if (not fc)
    return false;

  if (fc->dtype() != loco::DataType::FLOAT32)
    return false;

  if (fc->fusedActivationFunction() != luci::FusedActFunc::NONE)
    return false;

  auto weights = dynamic_cast<luci::CircleConst *>(fc->weights());
  if (not weights)
    return false;

  // Get add node
  auto fc_output = loco::succs(fc);
  if (fc_output.size() != 1)
    return false;

  auto add = dynamic_cast<luci::CircleAdd *>(*fc_output.begin());
  if (not add)
    return false;
  if (add->dtype() != loco::DataType::FLOAT32)
    return false;

  // Get addition
  auto addition = add->x() == fc ? dynamic_cast<luci::CircleConst *>(add->y())
                                 : dynamic_cast<luci::CircleConst *>(add->x());

  // Non-const addition
  if (not addition)
    return false;

  auto rank = addition->rank();
  // TODO Support scalar addition
  if (rank == 0)
    return false;

  for (uint32_t i = 0; i < rank - 1; i++)
  {
    if (addition->dim(i).value() != 1)
      return false;
  }
  // Check the last dimesion of addition is the same with the number of neurons of FC
  if (not(addition->dim(rank - 1) == weights->dim(0)))
    return false;

  auto bias = loco::must_cast<luci::CircleNode *>(fc->bias());

  // We only support (1) constant bias (2) no bias
  // If bias is neither (1) nor (2), it would be a feature map
  if (bias->opcode() != luci::CircleOpcode::CIRCLECONST and
      bias->opcode() != luci::CircleOpcode::CIRCLEOUTPUTEXCLUDE)
    return false;

  auto fused_bias = luci::clone(addition);

  // Add existing bias values
  if (auto const_bias = dynamic_cast<luci::CircleConst *>(fc->bias()))
  {
    assert(const_bias->dtype() == loco::DataType::FLOAT32);

    auto bias_size = fused_bias->size<loco::DataType::FLOAT32>();
    assert(bias_size == const_bias->size<loco::DataType::FLOAT32>());
    for (uint32_t i = 0; i < bias_size; i++)
      fused_bias->at<loco::DataType::FLOAT32>(i) += const_bias->at<loco::DataType::FLOAT32>(i);
  }

  // At this point, it is guarateed that fused_bias's shape is [1, 1, ..., N] or [N]
  // where N is weights->dim(0).
  // The shape is normalized to [N] to become the bias of FC
  fused_bias->rank(1);
  fused_bias->dim(0) = weights->dim(0);

  fc->bias(fused_bias);
  fc->fusedActivationFunction(add->fusedActivationFunction());

  // set origin
  luci::add_origin(fc, luci::get_origin(add));

  replace(add).with(fc);

  return true;
}

// Return qparam if it exists and its scale/zp's size is the same with len
// Return nullptr otherwise
luci::CircleQuantParam *get_qparam(luci::CircleNode *node, uint32_t len)
{
  if (node->quantparam() == nullptr)
    return nullptr;

  if (node->quantparam()->scale.size() != len)
    return nullptr;

  if (node->quantparam()->zerop.size() != len)
    return nullptr;

  return node->quantparam();
}

bool fuse_add_with_s16_fc(luci::CircleFullyConnected *fc)
{
  assert(fc);                                 // FIX_CALLER_UNLESS
  assert(fc->dtype() == loco::DataType::S16); // FIX_CALLER_UNLESS

  if (fc->fusedActivationFunction() != luci::FusedActFunc::NONE)
    return false;

  auto weights = dynamic_cast<luci::CircleConst *>(fc->weights());
  if (not weights)
    return false;

  auto fc_output = loco::succs(fc);
  // Fuse only when FC has a single successor (to avoid weight increase)
  if (fc_output.size() != 1)
    return false;

  auto add = dynamic_cast<luci::CircleAdd *>(*fc_output.begin());
  if (not add)
    return false;

  // Only support the same dtype with fc
  if (add->dtype() != loco::DataType::S16)
    return false;

  // Get addition
  auto addition = add->x() == fc ? dynamic_cast<luci::CircleConst *>(add->y())
                                 : dynamic_cast<luci::CircleConst *>(add->x());

  // Non-const addition
  if (not addition)
    return false;

  // Check addition dtype
  if (addition->dtype() != loco::DataType::S16)
    return false;

  auto rank = addition->rank();
  // TODO Support scalar addition
  if (rank == 0)
    return false;

  for (uint32_t i = 0; i < rank - 1; i++)
  {
    if (addition->dim(i).value() != 1)
      return false;
  }

  // Check the last dim of addition is the same with the output dim of weight
  const auto last_dim = addition->dim(rank - 1).value();
  if (last_dim != weights->dim(0).value())
    return false;

  auto bias = loco::must_cast<luci::CircleNode *>(fc->bias());

  // Only support (1) constant bias, or (2) no bias
  if (bias->opcode() != luci::CircleOpcode::CIRCLECONST and
      bias->opcode() != luci::CircleOpcode::CIRCLEOUTPUTEXCLUDE)
    return false;

  // If bias is const, its dtype must be s64
  if (bias->opcode() == luci::CircleOpcode::CIRCLECONST and bias->dtype() != loco::DataType::S64)
    return false;

  const auto addition_qparam = get_qparam(addition, last_dim);
  if (addition_qparam == nullptr)
    return false;

  std::vector<float> fp32_bias(last_dim);
  for (uint32_t i = 0; i < last_dim; i++)
  {
    auto scale = addition_qparam->scale.at(i);
    if (addition_qparam->zerop.at(i) != 0)
      return false; // FIX_ME_UNLESS

    auto val = addition->at<loco::DataType::S16>(i);
    fp32_bias[i] = val * scale;
  }

  // Add existing bias values
  if (auto const_bias = dynamic_cast<luci::CircleConst *>(bias))
  {
    const auto bias_qparam = get_qparam(const_bias, last_dim);
    if (bias_qparam == nullptr)
      return false;

    for (uint32_t i = 0; i < last_dim; i++)
    {
      auto scale = bias_qparam->scale.at(i);
      if (bias_qparam->zerop.at(i) != 0)
        return false; // FIX_ME_UNLESS

      auto val = const_bias->at<loco::DataType::S64>(i);
      fp32_bias[i] += val * scale;
    }
  }

  const auto add_qparam = get_qparam(add, 1);
  if (add_qparam == nullptr)
    return false;

  auto input = loco::must_cast<luci::CircleNode *>(fc->input());
  const auto input_qparam = get_qparam(input, 1);
  if (input_qparam == nullptr)
    return false;

  const auto weights_qparam = get_qparam(weights, last_dim);
  if (weights_qparam == nullptr)
    return false;

  auto fused_bias = luci::clone(addition);
  fused_bias->dtype(loco::DataType::S64);
  fused_bias->size<loco::DataType::S64>(last_dim);

  // The shape is normalized to [N] to become the bias of FC
  fused_bias->rank(1);
  fused_bias->dim(0) = last_dim;

  std::vector<float> new_bias_scale;
  for (uint32_t i = 0; i < last_dim; i++)
  {
    const auto input_scale = input_qparam->scale.at(0);
    const auto weight_scale = weights_qparam->scale.at(i);

    const float scale = input_scale * weight_scale;
    const float scale_inv = (scale == 0) ? 0 : 1.0 / scale;

    fused_bias->at<loco::DataType::S64>(i) =
      static_cast<int64_t>(std::round(fp32_bias.at(i) * scale_inv));

    new_bias_scale.push_back(scale);
  }
  std::vector<int64_t> new_bias_zerop(new_bias_scale.size(), 0);

  auto bias_qparam = std::make_unique<luci::CircleQuantParam>();
  {
    bias_qparam->scale = new_bias_scale;
    bias_qparam->zerop = new_bias_zerop;
  }

  fused_bias->quantparam(std::move(bias_qparam));

  // In-place update. This works because fc is guaranteed to have a single successor
  fc->bias(fused_bias);
  fc->fusedActivationFunction(add->fusedActivationFunction());

  auto qparam = std::make_unique<luci::CircleQuantParam>();
  {
    qparam->scale.push_back(add_qparam->scale.at(0));
    qparam->zerop.push_back(add_qparam->scale.at(0));
  }

  fc->quantparam(std::move(qparam));

  // set origin
  luci::add_origin(fc, luci::get_origin(add));

  replace(add).with(fc);

  return true;
}

} // namespace

namespace luci
{

bool FuseAddWithFullyConnectedPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto fc = dynamic_cast<luci::CircleFullyConnected *>(node);
    if (not fc)
      continue;

    switch (fc->dtype())
    {
      case loco::DataType::FLOAT32:
        if (fuse_add_with_fc(fc))
          changed = true;
        break;
      case loco::DataType::S16:
        if (fuse_add_with_s16_fc(fc))
          changed = true;
        break;
      default:
        break;
    }
  }

  return changed;
}

} // namespace luci
