/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/FuseMulWithConvPass.h"

#include "helpers/NodeFiller.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Profile/CircleNodeOrigin.h>
#include <luci/Service/CircleNodeClone.h>
#include <luci/Service/Nodes/CircleConst.h>

namespace
{

#define RETURN_FALSE_UNLESS(cond) \
  if (not(cond))                  \
    return false;

inline uint32_t cal_offset(const luci::CircleConst *node, const std::vector<uint32_t> &indices)
{
  // sanity check for node's rank
  assert(node != nullptr && node->rank() == 4);

  // sanity check for indices
  assert(indices.size() == 4);

  return indices[0] * node->dim(1).value() * node->dim(2).value() * node->dim(3).value() +
         indices[1] * node->dim(2).value() * node->dim(3).value() +
         indices[2] * node->dim(3).value() + indices[3];
}

/**
 *  Fuse Mul with Conv if possible
 *
 *  NOTE: In case mul is channewise constant, we can try to merge mul with nconv,
 *
 *  BEFORE
 *                    |
 *              [CircleConv2D] (no activation)
 *                    |
 *                  [Mul] (channel-wise/scalar constant)
 *                    |
 *
 *  AFTER
 *                    |
 *            [CircleConv2D] (with updated kernels, bias, and activation)
 *                    |
 *
 */

bool fuse_mul_with_conv(luci::CircleMul *mul)
{
  // sanity check
  RETURN_FALSE_UNLESS(mul->dtype() == loco::DataType::FLOAT32);

  luci::CircleConst *const_mul_operand = nullptr;
  luci::CircleConv2D *conv = nullptr;
  RETURN_FALSE_UNLESS(luci::fill(&const_mul_operand, &conv).with_commutative_args_of(mul));

  // sanity check
  RETURN_FALSE_UNLESS(conv->dtype() == loco::DataType::FLOAT32 &&
                      const_mul_operand->dtype() == loco::DataType::FLOAT32);

  //  NOTE for general activation function: S * Act(A * B) != Act(A*(SB))
  if (conv->fusedActivationFunction() != luci::FusedActFunc::NONE)
    return false;

  // check that const_mul_operand is channel-wise constant or just a scalar
  RETURN_FALSE_UNLESS(const_mul_operand->rank() == 4 || const_mul_operand->rank() == 1 ||
                      const_mul_operand->rank() == 0);

  std::vector<float> mul_values;
  if (const_mul_operand->rank() == 4)
  {
    // check channel-wise broadcasting
    RETURN_FALSE_UNLESS(const_mul_operand->dim(0).value() == 1 &&
                        const_mul_operand->dim(1).value() == 1 &&
                        const_mul_operand->dim(2).value() == 1);
  }
  else if (const_mul_operand->rank() == 1 || const_mul_operand->rank() == 0)
  {
    // sanity check
    RETURN_FALSE_UNLESS(const_mul_operand->size<loco::DataType::FLOAT32>() != 0);
  }

  mul_values.resize(const_mul_operand->size<loco::DataType::FLOAT32>());
  for (uint32_t idx = 0; idx < mul_values.size(); idx++)
  {
    mul_values[idx] = const_mul_operand->at<loco::DataType::FLOAT32>(idx);
  }

  // filter
  auto const conv_filter = dynamic_cast<luci::CircleConst *>(conv->filter());

  // sanity check
  RETURN_FALSE_UNLESS(conv_filter != nullptr && conv_filter->rank() == 4 &&
                      conv_filter->dtype() == loco::DataType::FLOAT32);

  auto const out_channels = conv_filter->dim(0).value();

  // multiplier is either channelwise constant or scalar
  RETURN_FALSE_UNLESS(out_channels == mul_values.size() || mul_values.size() == 1);

  // bias
  auto const conv_bias = dynamic_cast<luci::CircleConst *>(conv->bias());

  RETURN_FALSE_UNLESS(conv_bias == nullptr ||
                      (conv_bias->rank() == 1 && conv_bias->dim(0) == out_channels &&
                       conv_bias->dtype() == loco::DataType::FLOAT32));

  luci::CircleConst *fused_conv_filter = nullptr;
  {
    // fused filter
    fused_conv_filter = luci::clone(conv_filter);
    // set values of conv filter multiplied by constant channel-wise
    for (uint32_t out_chan = 0; out_chan < out_channels; out_chan++)
    {
      // for scalar - first element, otherwise - channelwise
      float mult = mul_values[out_chan % mul_values.size()];
      for (uint32_t out_height = 0; out_height < fused_conv_filter->dim(1).value(); out_height++)
      {
        for (uint32_t out_width = 0; out_width < fused_conv_filter->dim(2).value(); out_width++)
        {
          for (uint32_t in_chan = 0; in_chan < fused_conv_filter->dim(3).value(); in_chan++)
          {
            std::vector<uint32_t> indices = {out_chan, out_height, out_width, in_chan};
            auto const data =
              conv_filter->at<loco::DataType::FLOAT32>(cal_offset(conv_filter, indices));
            fused_conv_filter->at<loco::DataType::FLOAT32>(cal_offset(fused_conv_filter, indices)) =
              mult * data;
          }
        }
      }
    }
    fused_conv_filter->name(conv_filter->name() + "/FusedMul");
    luci::add_origin(fused_conv_filter, luci::get_origin(conv_filter));
  }

  luci::CircleConst *fused_conv_bias = nullptr;
  if (conv_bias != nullptr)
  {
    // fused bias
    fused_conv_bias = luci::clone(conv_bias);
    // update bias values
    for (uint32_t c = 0; c < conv_bias->size<loco::DataType::FLOAT32>(); c++)
    {
      // for scalar - first element, otherwise - channelwise
      float mult = mul_values[c % mul_values.size()];
      auto const data = conv_bias->at<loco::DataType::FLOAT32>(c);
      fused_conv_bias->at<loco::DataType::FLOAT32>(c) = mult * data;
    }

    fused_conv_bias->name(conv_bias->name() + "/FusedMul");
    luci::add_origin(fused_conv_bias, luci::get_origin(conv_bias));
  }

  // Configure new CircleConv2D operation.
  auto *fused_conv = loco::must_cast<luci::CircleConv2D *>(luci::clone_node(conv, mul->graph()));
  fused_conv->input(conv->input());
  fused_conv->filter(fused_conv_filter);
  fused_conv->bias(fused_conv_bias);
  fused_conv->name(conv->name() + "/FusedMul");
  fused_conv->fusedActivationFunction(mul->fusedActivationFunction());
  luci::add_origin(fused_conv,
                   luci::composite_origin({luci::get_origin(conv), luci::get_origin(mul)}));

  // Replace old mul operation with new fused_conv with updated kernel/bias
  replace(mul).with(fused_conv);

  return true;
}

} // namespace

namespace luci
{

bool FuseMulWithConvPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto mul = dynamic_cast<luci::CircleMul *>(node);
    if (not mul)
      continue;

    if (fuse_mul_with_conv(mul))
      changed = true;
  }

  return changed;
}

} // namespace luci
