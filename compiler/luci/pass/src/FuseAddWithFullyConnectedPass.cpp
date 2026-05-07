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

#include "helpers/NodeFiller.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Service/Nodes/CircleConst.h>
#include <luci/Profile/CircleNodeOrigin.h>

#include <cmath>

namespace
{

#define RETURN_FALSE_UNLESS(cond) \
  if (not(cond))                  \
    return false;

struct PatternNodes
{
  luci::CircleFullyConnected *fc = nullptr;
  // addition must be const
  luci::CircleConst *addition = nullptr;
  luci::CircleConst *weights = nullptr;
  luci::CircleNode *bias = nullptr;
};

bool fc_with_add_pattern_check(const loco::DataType dtype, const luci::CircleAdd &add,
                               PatternNodes &nodes)
{
  RETURN_FALSE_UNLESS(add.dtype() == dtype);

  RETURN_FALSE_UNLESS(luci::fill(&nodes.fc, &nodes.addition).with_commutative_args_of(&add));

  // Check if fc has only one successor to limit possible weights size increase.
  RETURN_FALSE_UNLESS(loco::succs(nodes.fc).size() == 1);
  RETURN_FALSE_UNLESS(nodes.fc->dtype() == dtype);
  RETURN_FALSE_UNLESS(nodes.fc->fusedActivationFunction() == luci::FusedActFunc::NONE);

  nodes.weights = dynamic_cast<luci::CircleConst *>(nodes.fc->weights());
  RETURN_FALSE_UNLESS(nodes.weights);

  RETURN_FALSE_UNLESS(nodes.addition->dtype() == dtype);

  auto rank = (nodes.addition)->rank();
  // TODO Support scalar addition
  RETURN_FALSE_UNLESS(rank != 0);

  for (uint32_t i = 0; i < rank - 1; i++)
  {
    RETURN_FALSE_UNLESS(nodes.addition->dim(i).value() == 1);
  }
  // Check the last dimesion of addition is the same with the number of neurons of FC
  RETURN_FALSE_UNLESS(nodes.addition->dim(rank - 1) == nodes.weights->dim(0));

  // We only support (1) constant bias (2) no bias
  // If bias is neither (1) nor (2), it would be a feature map
  nodes.bias = loco::must_cast<luci::CircleNode *>(nodes.fc->bias());
  RETURN_FALSE_UNLESS(nodes.bias->opcode() == luci::CircleOpcode::CIRCLECONST or
                      nodes.bias->opcode() == luci::CircleOpcode::CIRCLEOUTPUTEXCLUDE);

  return true;
}

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
bool fuse_add_with_fc(luci::CircleAdd *add)
{
  PatternNodes nodes;

  RETURN_FALSE_UNLESS(fc_with_add_pattern_check(loco::DataType::FLOAT32, *add, nodes));

  auto fused_bias = luci::clone(nodes.addition);

  // Add existing bias values
  if (auto const_bias = dynamic_cast<luci::CircleConst *>(nodes.fc->bias()))
  {
    RETURN_FALSE_UNLESS(const_bias->dtype() == loco::DataType::FLOAT32);

    auto bias_size = fused_bias->size<loco::DataType::FLOAT32>();
    RETURN_FALSE_UNLESS(bias_size == const_bias->size<loco::DataType::FLOAT32>());
    for (uint32_t i = 0; i < bias_size; i++)
      fused_bias->at<loco::DataType::FLOAT32>(i) += const_bias->at<loco::DataType::FLOAT32>(i);
  }

  // At this point, it is guarateed that fused_bias's shape is [1, 1, ..., N] or [N]
  // where N is weights->dim(0).
  // The shape is normalized to [N] to become the bias of FC
  fused_bias->rank(1);
  fused_bias->dim(0) = nodes.weights->dim(0);

  nodes.fc->bias(fused_bias);
  nodes.fc->fusedActivationFunction(add->fusedActivationFunction());

  // set origin
  luci::add_origin(nodes.fc, luci::get_origin(add));

  replace(add).with(nodes.fc);

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

bool fuse_add_with_s16_fc(luci::CircleAdd *add)
{
  PatternNodes nodes;

  RETURN_FALSE_UNLESS(fc_with_add_pattern_check(loco::DataType::S16, *add, nodes));

  // If bias is const, its dtype must be s64
  RETURN_FALSE_UNLESS(nodes.bias->opcode() == luci::CircleOpcode::CIRCLECONST and
                      nodes.bias->dtype() == loco::DataType::S64);

  const auto last_dim = nodes.addition->dim(nodes.addition->rank() - 1).value();

  const auto addition_qparam = get_qparam(nodes.addition, last_dim);
  RETURN_FALSE_UNLESS(addition_qparam);

  std::vector<float> fp32_bias(last_dim);
  for (uint32_t i = 0; i < last_dim; i++)
  {
    auto scale = addition_qparam->scale.at(i);
    RETURN_FALSE_UNLESS(addition_qparam->zerop.at(i) == 0);

    auto val = nodes.addition->at<loco::DataType::S16>(i);
    fp32_bias[i] = val * scale;
  }

  // Add existing bias values
  if (auto const_bias = dynamic_cast<luci::CircleConst *>(nodes.bias))
  {
    const auto bias_qparam = get_qparam(const_bias, last_dim);
    RETURN_FALSE_UNLESS(bias_qparam);

    for (uint32_t i = 0; i < last_dim; i++)
    {
      auto scale = bias_qparam->scale.at(i);
      RETURN_FALSE_UNLESS(bias_qparam->zerop.at(i) == 0);

      auto val = const_bias->at<loco::DataType::S64>(i);
      fp32_bias[i] += val * scale;
    }
  }

  const auto add_qparam = get_qparam(add, 1);
  RETURN_FALSE_UNLESS(add_qparam);

  auto input = loco::must_cast<luci::CircleNode *>(nodes.fc->input());
  const auto input_qparam = get_qparam(input, 1);
  RETURN_FALSE_UNLESS(input_qparam);

  const auto weights_qparam = get_qparam(nodes.weights, last_dim);
  RETURN_FALSE_UNLESS(weights_qparam);

  auto fused_bias = luci::clone(nodes.addition);
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
  nodes.fc->bias(fused_bias);
  nodes.fc->fusedActivationFunction(add->fusedActivationFunction());

  auto qparam = std::make_unique<luci::CircleQuantParam>();
  {
    qparam->scale.push_back(add_qparam->scale.at(0));
    qparam->zerop.push_back(add_qparam->scale.at(0));
  }

  nodes.fc->quantparam(std::move(qparam));

  // set origin
  luci::add_origin(nodes.fc, luci::get_origin(add));

  replace(add).with(nodes.fc);

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
    auto add = dynamic_cast<luci::CircleAdd *>(node);
    if (not add)
      continue;

    switch (add->dtype())
    {
      case loco::DataType::FLOAT32:
        if (fuse_add_with_fc(add))
          changed = true;
        break;
      case loco::DataType::S16:
        if (fuse_add_with_s16_fc(add))
          changed = true;
        break;
      default:
        break;
    }
  }

  return changed;
}

} // namespace luci
