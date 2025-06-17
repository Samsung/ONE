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
#include <luci/Service/Nodes/CircleConst.h>
#include <luci/Profile/CircleNodeOrigin.h>

namespace
{

#define RETURN_FALSE_UNLESS(cond) \
  if (not(cond))                  \
    return false;

inline bool is_effectively_scalar_shape(const luci::CircleConst *node)
{
  if (node->size<loco::DataType::FLOAT32>() != 1)
    return false;
  auto const rank = node->rank();
  for (uint32_t i = 0; i < rank; i++)
    if (node->dim(i).value() != 1)
      return false;
  return true;
}

inline bool is_single_element(const luci::CircleConst *node)
{
  if (is_effectively_scalar_shape(node))
    return true;
  return ((node->rank() == 1 || node->rank() == 0) && node->size<loco::DataType::FLOAT32>() == 1);
}

inline void update_with_single_element(luci::CircleConst *fused_node,
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
  // Single element multiplication:
  if (is_single_element(multiplication))
  {
    update_with_single_element(fused_weights, multiplication);
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
  // Single element multiplication:
  if (is_single_element(multiplication))
  {
    update_with_single_element(fused_bias, multiplication);
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
bool fuse_mul_with_fc(luci::CircleMul *mul)
{
  // Sanity check:
  RETURN_FALSE_UNLESS(mul);
  // Allow Mul node only with FLOAT32 data type:
  RETURN_FALSE_UNLESS(mul->dtype() == loco::DataType::FLOAT32);
  // Check if any FC node connects to Mul.
  // Find the pattern of Mul(FC, CircleConst):
  luci::CircleFullyConnected *fc = nullptr;
  luci::CircleConst *multiplication = nullptr;
  RETURN_FALSE_UNLESS(luci::fill(&fc, &multiplication).with_commutative_args_of(mul));
  /**
   *  Make sure that FullyConnected has only one successor.
   *
   *  If the FullyConnected output is connected to more nodes,
   *  this pass will replace node with new fused FullyConnected.
   *  Thus pass success will only introduce extra FullyConnected
   *  without reducing overall number of nodes.
   *  Which tends to increase model's size and degrades model's performance.
   *  Thus one successor is required to benefit from this pass.
   *
   *  Example graph that illustrates the described scenario:
   *
   *  BEFORE
   *                |
   *      [CircleFullyConnected]
   *                |
   *        +-------+----------------+
   *        |                        |
   *        |                        |
   *  [Other Node]              [CircleMul]
   *        |                        |
   *
   *  AFTER
   *                |
   *                +-----------------------+
   *                |                       |
   *                |                       |
   *      [CircleFullyConnected]            |
   *                |                       |
   *        +-------+                       |
   *        |                               |
   *        |                               |
   *  [Other Node]       [New CircleFullyConnected Fused with Mul]
   *        |                               |
   *
   */
  RETURN_FALSE_UNLESS(loco::succs(fc).size() == 1);
  // Allow only FLOAT32 data type:
  RETURN_FALSE_UNLESS(fc->dtype() == loco::DataType::FLOAT32);
  // Allow only without activation functions as values are going to
  // be multiplied before activation function.
  RETURN_FALSE_UNLESS(fc->fusedActivationFunction() == luci::FusedActFunc::NONE);
  // Check for weights being Constant:
  auto weights = dynamic_cast<luci::CircleConst *>(fc->weights());
  RETURN_FALSE_UNLESS(weights);
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
    RETURN_FALSE_UNLESS(multiplication->dim(rank - 1) == weights->dim(0) ||
                        is_effectively_scalar_shape(multiplication));
  }
  // 1-D or scalar case:
  else if (multiplication->rank() == 1)
  {
    RETURN_FALSE_UNLESS(multiplication->size<loco::DataType::FLOAT32>() == 1 ||
                        multiplication->size<loco::DataType::FLOAT32>() == weights->dim(0));
  }
  else if (multiplication->rank() == 0)
  {
    RETURN_FALSE_UNLESS(multiplication->size<loco::DataType::FLOAT32>() == 1);
  }

  // Only supports:
  // (1) constant bias
  // (2) no bias
  auto bias = loco::must_cast<luci::CircleNode *>(fc->bias());
  if (bias->opcode() == luci::CircleOpcode::CIRCLECONST)
  {
    // Create new bias to be updated with values:
    auto const_bias = dynamic_cast<luci::CircleConst *>(fc->bias());
    RETURN_FALSE_UNLESS(const_bias)
    RETURN_FALSE_UNLESS(const_bias->dtype() == loco::DataType::FLOAT32);
    // Create new bias with updated values and replace:
    auto fused_bias = gen_fused_bias(const_bias, multiplication);
    fc->bias(fused_bias);
  }
  else if (bias->opcode() != luci::CircleOpcode::CIRCLEOUTPUTEXCLUDE)
  {
    return false;
  }

  // Create new weights with updated values and replace:
  auto fused_weights = gen_fused_weights(weights, multiplication);
  fc->weights(fused_weights);

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
    if (auto mul = dynamic_cast<luci::CircleMul *>(node))
    {
      if (fuse_mul_with_fc(mul))
        changed = true;
    }
  }

  return changed;
}

} // namespace luci
