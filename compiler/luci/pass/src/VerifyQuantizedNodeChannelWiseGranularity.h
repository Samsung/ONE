/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_VERIFY_QUANTIZED_NODE_CHANNELWISE_GRANULARITY_H__
#define __LUCI_VERIFY_QUANTIZED_NODE_CHANNELWISE_GRANULARITY_H__

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeVisitor.h>
#include <luci/Pass/QuantizationParameters.h>

using Granularity = luci::QuantizationGranularity;

// This macro is undef at the end of the file
#define RETURN_FALSE_UNLESS(ARG) \
  if (not(ARG))                  \
  {                              \
    return false;                \
  }

namespace luci
{

/**
 * @brief Verify the granualrity of channel-wise quantized node
 * @details
 *
 * Targets to verify
 * - node's output (i.e., node itself)
 * - node's inputs
 */
struct VerifyQuantizedNodeChannelWiseGranularity final : public luci::CircleNodeVisitor<bool>
{
private:
  bool is_lwq(const loco::Node *node)
  {
    auto circle_node = loco::must_cast<const luci::CircleNode *>(node);

    if (circle_node->quantparam() == nullptr)
      return false;

    if (circle_node->quantparam()->scale.size() != 1)
      return false;

    if (circle_node->quantparam()->zerop.size() != 1)
      return false;

    return true;
  }

  uint32_t rank(const loco::Node *node)
  {
    auto circle_node = loco::must_cast<const luci::CircleNode *>(node);
    return circle_node->rank();
  }

  bool is_cwq_const(const loco::Node *node, uint32_t channel_dim)
  {
    auto circle_node = loco::must_cast<const luci::CircleConst *>(node);

    assert(channel_dim < circle_node->rank()); // FIX_CALLER_UNLESS
    auto channel_size = circle_node->dim(channel_dim).value();

    if (circle_node->quantparam() == nullptr)
      return false;

    if (circle_node->quantparam()->quantized_dimension != static_cast<int32_t>(channel_dim))
      return false;

    if (circle_node->quantparam()->scale.size() != channel_size)
      return false;

    if (circle_node->quantparam()->zerop.size() != channel_size)
      return false;

    return true;
  }

private:
  bool visit(const luci::CircleConv2D *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node))
    RETURN_FALSE_UNLESS(is_lwq(node->input()))
    RETURN_FALSE_UNLESS(is_cwq_const(node->filter(), 0))
    RETURN_FALSE_UNLESS(is_cwq_const(node->bias(), rank(node->bias()) - 1))
    return true;
  }

  bool visit(const luci::CircleDepthwiseConv2D *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node))
    RETURN_FALSE_UNLESS(is_lwq(node->input()))
    RETURN_FALSE_UNLESS(is_cwq_const(node->filter(), 3))
    RETURN_FALSE_UNLESS(is_cwq_const(node->bias(), rank(node->bias()) - 1))
    return true;
  }

  bool visit(const luci::CircleInstanceNorm *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node))
    RETURN_FALSE_UNLESS(is_lwq(node->input()))
    RETURN_FALSE_UNLESS(is_cwq_const(node->gamma(), rank(node->gamma()) - 1))
    RETURN_FALSE_UNLESS(is_cwq_const(node->beta(), rank(node->beta()) - 1))
    return true;
  }

  bool visit(const luci::CirclePad *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node))
    RETURN_FALSE_UNLESS(is_lwq(node->input()))
    return true;
  }

  bool visit(const luci::CirclePRelu *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node))
    RETURN_FALSE_UNLESS(is_lwq(node->input()))
    RETURN_FALSE_UNLESS(is_cwq_const(node->alpha(), rank(node->alpha()) - 1))
    return true;
  }

  bool visit(const luci::CircleTransposeConv *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node))
    RETURN_FALSE_UNLESS(is_lwq(node->outBackprop()))
    RETURN_FALSE_UNLESS(is_cwq_const(node->filter(), 0))
    luci::CircleConst *bias = dynamic_cast<luci::CircleConst *>(node->bias());
    if (bias != nullptr)
      RETURN_FALSE_UNLESS(is_cwq_const(node->bias(), rank(node->bias()) - 1))

    return true;
  }

  bool visit(const luci::CircleFullyConnected *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node))
    RETURN_FALSE_UNLESS(is_lwq(node->input()))
    RETURN_FALSE_UNLESS(is_cwq_const(node->weights(), 0))
    RETURN_FALSE_UNLESS(is_cwq_const(node->bias(), rank(node->bias()) - 1))
    return true;
  }

  bool visit(const luci::CircleAdd *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node));
    RETURN_FALSE_UNLESS(is_lwq(node->x()));
    RETURN_FALSE_UNLESS(is_lwq(node->y()));
    return true;
  }

  bool visit(const luci::CircleAveragePool2D *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node));
    RETURN_FALSE_UNLESS(is_lwq(node->value()));
    return true;
  }

  bool visit(const luci::CircleLogicalOr *)
  {
    // Logical OR has bool-type inputs and output
    // Nothing to be checked
    return true;
  }

  bool visit(const luci::CircleMaxPool2D *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node));
    RETURN_FALSE_UNLESS(is_lwq(node->value()));
    return true;
  }

  bool visit(const luci::CircleMean *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node));
    RETURN_FALSE_UNLESS(is_lwq(node->input()));
    return true;
  }

  bool visit(const luci::CircleMul *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node));
    RETURN_FALSE_UNLESS(is_lwq(node->x()));
    RETURN_FALSE_UNLESS(is_lwq(node->y()));
    return true;
  }

  bool visit(const luci::CircleNotEqual *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node->x()));
    RETURN_FALSE_UNLESS(is_lwq(node->y()));
    return true;
  }

  bool visit(const luci::CircleRelu *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node));
    RETURN_FALSE_UNLESS(is_lwq(node->features()));
    return true;
  }

  bool visit(const luci::CircleReshape *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node))
    RETURN_FALSE_UNLESS(is_lwq(node->tensor()));
    return true;
  }

  bool visit(const luci::CircleLogistic *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node));
    RETURN_FALSE_UNLESS(is_lwq(node->x()));
    return true;
  }

  bool visit(const luci::CircleSoftmax *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node));
    RETURN_FALSE_UNLESS(is_lwq(node->logits()));
    return true;
  }

  bool visit(const luci::CircleSlice *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node));
    RETURN_FALSE_UNLESS(is_lwq(node->input()));
    return true;
  }

  bool visit(const luci::CircleArgMax *node)
  {
    // node's output is index, thus not quantized
    RETURN_FALSE_UNLESS(is_lwq(node->input()));
    return true;
  }

  bool visit(const luci::CircleTanh *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node));
    RETURN_FALSE_UNLESS(is_lwq(node->x()));
    return true;
  }

  bool visit(const luci::CircleTranspose *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node));
    RETURN_FALSE_UNLESS(is_lwq(node->a()));
    return true;
  }

  bool visit(const luci::CircleFloor *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node));
    RETURN_FALSE_UNLESS(is_lwq(node->x()));
    return true;
  }

  bool visit(const luci::CircleGreater *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node->x()));
    RETURN_FALSE_UNLESS(is_lwq(node->y()));
    return true;
  }

  bool visit(const luci::CircleGreaterEqual *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node->x()));
    RETURN_FALSE_UNLESS(is_lwq(node->y()));
    return true;
  }

  bool visit(const luci::CircleDiv *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node));
    RETURN_FALSE_UNLESS(is_lwq(node->x()));
    RETURN_FALSE_UNLESS(is_lwq(node->y()));
    return true;
  }

  bool visit(const luci::CircleFloorDiv *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node));
    RETURN_FALSE_UNLESS(is_lwq(node->x()));
    RETURN_FALSE_UNLESS(is_lwq(node->y()));
    return true;
  }

  // TODO: Implement more Ops

  bool visit(const luci::CircleNode *) { return true; }
};

} // namespace luci

#undef RETURN_FALSE_UNLESS

#endif // __LUCI_VERIFY_QUANTIZED_NODE_CHANNELWISE_GRANULARITY_H__
