/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/FuseMulWithFullyConnectedPass.h"

#include "helpers/NodeFiller.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Profile/CircleNodeOrigin.h>
#include <luci/Service/CircleNodeClone.h>
#include <luci/Service/Nodes/CircleConst.h>

namespace luci
{

namespace
{
/**
 *  Fuse Mul with FullyConnected if possible
 *
 *  BEFORE
 *                    |
 *              [FullyConnected] (no activation)
 *                    |
 *                  [Mul] (channel-wise/scalar constant)
 *                    |
 *
 *  AFTER
 *                    |
 *            [FullyConnected] (with updated kernels, bias)
 *                    |
 *
 */
bool fuse_mul_with_fully_connected(luci::CircleMul *mul)
{
  if (mul->fusedActivationFunction() != luci::FusedActFunc::NONE)
    return false;

  luci::CircleFullyConnected *fc = nullptr;
  luci::CircleConst *mul_const = nullptr;
  if (not luci::fill(&fc, &mul_const).with_args_of(mul))
    return false;

  if (fc->fusedActivationFunction() != luci::FusedActFunc::NONE)
    return false;

  if (mul_const->dtype() != loco::DataType::FLOAT32)
    return false;

  if (fc->dtype() != loco::DataType::FLOAT32)
    return false;

  // check that mul_const is a scalar
  if (mul_const->rank() != 0 and mul_const->rank() != 1)
  {
    // Otherwise check that all dims is equal to 1 except last one
    for (uint32_t i = 0; i < mul_const->rank() - 1; ++i)
    {
      if (mul_const->dim(i).value() != 1)
        return false;
    }
  }

  luci::CircleNode *fc_input = dynamic_cast<luci::CircleNode *>(fc->input());
  luci::CircleConst *fc_weight = dynamic_cast<luci::CircleConst *>(fc->weights());
  luci::CircleConst *fc_bias = dynamic_cast<luci::CircleConst *>(fc->bias());

  if (fc_weight == nullptr)
    return false;

  if (fc_weight->dtype() != loco::DataType::FLOAT32)
    return false;

  if (fc_weight->rank() != 2)
    return false;

  // check size is equal to 1 or number of rows of fully connected weights
  if (mul_const->size<loco::DataType::FLOAT32>() != 1 and
      mul_const->size<loco::DataType::FLOAT32>() != fc_weight->dim(0).value())
    return false;

  if (fc_bias != nullptr)
  {
    if (fc_bias->rank() != 1 or fc_bias->dtype() != loco::DataType::FLOAT32)
      return false;
  }

  auto mult_const_size = mul_const->size<loco::DataType::FLOAT32>();

  luci::CircleConst *fused_fc_weight = nullptr;
  {
    fused_fc_weight = luci::clone(fc_weight);
    for (uint32_t i = 0; i < fc_weight->dim(0).value(); ++i)
    {
      float mult = mult_const_size == 1 ? mul_const->at<loco::DataType::FLOAT32>(0)
                                        : mul_const->at<loco::DataType::FLOAT32>(i);
      for (uint32_t j = 0; j < fc_weight->dim(1).value(); ++j)
      {
        fc_weight->at<loco::DataType::FLOAT32>(i * fc_weight->dim(0).value() + j) *= mult;
      }
    }
    fused_fc_weight->name(fused_fc_weight->name() + "/FusedMul");
    luci::add_origin(fused_fc_weight, luci::get_origin(fc_weight));
  }

  luci::CircleConst *fused_fc_bias = nullptr;
  if (fc_bias != nullptr)
  {
    // fused bias
    fused_fc_bias = luci::clone(fc_bias);
    // update bias values
    for (uint32_t c = 0; c < fc_bias->size<loco::DataType::FLOAT32>(); c++)
    {
      float mult = mult_const_size == 1 ? mul_const->at<loco::DataType::FLOAT32>(0)
                                        : mul_const->at<loco::DataType::FLOAT32>(c);
      fused_fc_bias->at<loco::DataType::FLOAT32>(c) *= mult;
    }

    fused_fc_bias->name(fc_bias->name() + "/FusedMul");
    luci::add_origin(fused_fc_bias, luci::get_origin(fc_bias));
  }

  // Configure new FullyConnected operation.
  auto *fused_fc =
    loco::must_cast<luci::CircleFullyConnected *>(luci::clone_node(fc, mul->graph()));
  fused_fc->input(fc_input);
  fused_fc->weights(fused_fc_weight);
  if (fused_fc_bias != nullptr)
  {
    fused_fc->bias(fused_fc_bias);
  }
  else
  {
    auto bias_output = mul->graph()->nodes()->create<luci::CircleOutputExclude>();
    fused_fc->bias(bias_output);
  }
  fused_fc->name(fc->name() + "/FusedMul");
  fused_fc->fusedActivationFunction(mul->fusedActivationFunction());
  luci::add_origin(fused_fc, luci::composite_origin({luci::get_origin(fc), luci::get_origin(mul)}));

  // Replace old mul operation with new fused_conv with updated kernel/bias
  replace(mul).with(fused_fc);

  return true;
}

} // namespace

bool FuseMulWithFullyConnectedPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto mul = dynamic_cast<luci::CircleMul *>(node);
    if (not mul)
      continue;

    if (fuse_mul_with_fully_connected(mul))
      changed = true;
  }

  return changed;
}

} // namespace luci
