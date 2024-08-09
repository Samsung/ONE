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

#include <luci/IR/CircleNodes.h>
#include <luci/Service/Nodes/CircleConst.h>
#include <luci/Profile/CircleNodeOrigin.h>

#include <cmath>

namespace
{

#define RETURN_FALSE_UNLESS(cond) \
  if (not(cond))                  \
    return false;


inline bool is_scalar(const luci::CircleConst *node)
{
  return ((node->rank() == 1 || node->rank() == 0) && node->size<loco::DataType::FLOAT32>() == 1);
}

inline void update_with_scalar(luci::CircleConst *fused_node,
                               const luci::CircleConst *multiplication)
{
  for (uint32_t i = 0; i < fused_node->size<loco::DataType::FLOAT32>(); i++)
  {
    fused_node->at<loco::DataType::FLOAT32>(i) *= multiplication->at<loco::DataType::FLOAT32>(0);
  }
}

luci::CircleConst *gen_fused_weights(luci::CircleConst *weights,
                                     const luci::CircleConst *multiplication)
{
  auto fused_weights = luci::clone(weights);
  // Scalar multiplication:
  if (is_scalar(multiplication))
  {
    update_with_scalar(fused_weights, multiplication);
  }
  // N-size multiplication:
  else
  {
    // Go along channels, multiplication size is ensured to be compatible with channels.
    auto count = fused_weights->dim(0).value();
    auto size = fused_weights->dim(fused_weights->rank() - 1).value();
    float val;
    for (uint32_t c = 0; c < count; c++)
    {
      val = multiplication->at<loco::DataType::FLOAT32>(c);
      for (uint32_t i = 0; i < size; i++)
      {
        fused_weights->at<loco::DataType::FLOAT32>(c * size + i) *= val;
      }
    }
  }
  return fused_weights;
}

luci::CircleConst *gen_fused_bias(luci::CircleConst *bias, const luci::CircleConst *multiplication)
{
  auto fused_bias = luci::clone(bias);
  // Scalar multiplication:
  if (is_scalar(multiplication))
  {
    update_with_scalar(fused_bias, multiplication);
  }
  // N-size multiplication:
  else
  {
    // Go along channels, multiplication size is ensured to be compatible with channels.
    for (uint32_t i = 0; i < fused_bias->size<loco::DataType::FLOAT32>(); i++)
    {
      fused_bias->at<loco::DataType::FLOAT32>(i) *= multiplication->at<loco::DataType::FLOAT32>(i);
    }
  }
  return fused_bias;
}

/**
 *  Fuse Mul to FullyConnected if the multiplied value is a channel(last dimension)-wise constant
 *
 *  BEFORE
 *                |
 *      [CircleFullyConnected]
 *                |
 *           [CircleMul]
 *                |
 *
 *  AFTER
 *                |
 *       [CircleFullyConnected]   [CircleMul] (dead)
 *                |
 *
 */
bool fuse_mul_with_fc(luci::CircleFullyConnected *fc)
{
  // Sanity check:
  RETURN_FALSE_UNLESS(fc);
  // Allow only FLOAT32 data type:
  RETURN_FALSE_UNLESS(fc->dtype() == loco::DataType::FLOAT32);
  // Allow only without activation functions as values are going to
  // be multiplied before activation function.
  RETURN_FALSE_UNLESS(fc->fusedActivationFunction() == luci::FusedActFunc::NONE);
  // Check for weights being Constant:
  auto weights = dynamic_cast<luci::CircleConst *>(fc->weights());
  RETURN_FALSE_UNLESS(weights);
  // Get Mul node:
  auto fc_output = loco::succs(fc);
  // Make sure that FullyConnected has only one child:
  RETURN_FALSE_UNLESS(fc_output.size() == 1);
  auto mul = dynamic_cast<luci::CircleMul *>(*fc_output.begin());
  RETURN_FALSE_UNLESS(mul);
  // Allow Mul node only with FLOAT32 data type:
  RETURN_FALSE_UNLESS(mul->dtype() == loco::DataType::FLOAT32);
  // Get multiplication Constant (here: the second input besides weights):
  auto multiplication = mul->x() == fc ? dynamic_cast<luci::CircleConst *>(mul->y())
                                       : dynamic_cast<luci::CircleConst *>(mul->x());
  RETURN_FALSE_UNLESS(multiplication);
  // Get rank of multiplication:
  auto rank = multiplication->rank();
  // Check that all dimensions are ones, checks broadcast capabilites.
  // Last dimesion of multiplication must be compatible with FC.
  // N-D case (N>1):
  if (multiplication->rank() > 1)
  {
    // Check channel-wise broadcasting:
    for (uint32_t i = 0; i < rank - 1; i++)
      RETURN_FALSE_UNLESS(multiplication->dim(i).value() == 1);
    // Check the last dimesion of Mul is the same with the first dimension of FullyConnected
    RETURN_FALSE_UNLESS(multiplication->dim(rank - 1) == weights->dim(0));
  }
  // Scalar case:
  else if (multiplication->rank() == 1 || multiplication->rank() == 0)
  {
    RETURN_FALSE_UNLESS(multiplication->size<loco::DataType::FLOAT32>() != 0);
  }

  // Only supports:
  // (1) constant bias
  // (2) no bias
  auto bias = loco::must_cast<luci::CircleNode *>(fc->bias());
  RETURN_FALSE_UNLESS(bias->opcode() == luci::CircleOpcode::CIRCLECONST or
                      bias->opcode() == luci::CircleOpcode::CIRCLEOUTPUTEXCLUDE)
  // Create new bias to be updated with values:
  auto const_bias = dynamic_cast<luci::CircleConst *>(fc->bias());
  RETURN_FALSE_UNLESS(const_bias)
  RETURN_FALSE_UNLESS(const_bias->dtype() == loco::DataType::FLOAT32);

  // Create new weights and bias with updated values:
  auto fused_bias = gen_fused_bias(const_bias, multiplication);
  auto fused_weights = gen_fused_weights(weights, multiplication);

  // Replace weights and bias:
  fc->weights(fused_weights);
  fc->bias(fused_bias);

  // Set origin and copy Activation Function if exisitng:
  fc->fusedActivationFunction(mul->fusedActivationFunction());
  luci::add_origin(fc, luci::get_origin(mul));

  replace(mul).with(fc);

  return true;
}

} // namespace

namespace luci
{

bool FuseMulWithFullyConnectedPass::run(loco::Graph *g)
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
        if (fuse_mul_with_fc(fc))
          changed = true;
        break;
      default:
        break;
    }
  }

  return changed;
}

} // namespace luci
