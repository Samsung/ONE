/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_VERIFY_QUANTIZED_NODE_GRANULARITY_H__
#define __LUCI_VERIFY_QUANTIZED_NODE_GRANULARITY_H__

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeVisitor.h>
#include <luci/Pass/QuantizationParameters.h>

#include <memory>

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
 * @brief Verify the granualrity of quantized node
 * @details
 *
 * Targets to verify
 * - node's output (i.e., node itself)
 * - node's inputs
 */
class VerifyQuantizedNodeGranularity : public luci::CircleNodeVisitor<bool>
{
public:
  static std::shared_ptr<VerifyQuantizedNodeGranularity> create(Granularity granularity);

protected:
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

private:
  virtual bool visit(const luci::CircleConv2D *node) = 0;

  bool visit(const luci::CircleConcatenation *node)
  {
    // Skip granularity check for concatenation of indices
    if (node->dtype() == loco::DataType::S32 or node->dtype() == loco::DataType::S64)
      return true;

    RETURN_FALSE_UNLESS(is_lwq(node))
    for (uint32_t i = 0; i < node->numValues(); i++)
    {
      RETURN_FALSE_UNLESS(is_lwq(node->values(i)));
    }
    return true;
  }

  bool visit(const luci::CircleDepthToSpace *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node))
    RETURN_FALSE_UNLESS(is_lwq(node->input()))
    return true;
  }

  virtual bool visit(const luci::CircleDepthwiseConv2D *node) = 0;

  virtual bool visit(const luci::CircleInstanceNorm *node) = 0;

  bool visit(const luci::CirclePack *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node))
    for (uint32_t i = 0; i < node->values_count(); i++)
    {
      RETURN_FALSE_UNLESS(is_lwq(node->values(i)));
    }
    return true;
  }

  bool visit(const luci::CirclePad *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node))
    RETURN_FALSE_UNLESS(is_lwq(node->input()))
    return true;
  }

  bool visit(const luci::CirclePadV2 *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node))
    RETURN_FALSE_UNLESS(is_lwq(node->input()))
    RETURN_FALSE_UNLESS(is_lwq(node->constant_values()))
    return true;
  }

  bool visit(const luci::CircleMirrorPad *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node))
    RETURN_FALSE_UNLESS(is_lwq(node->input()))
    return true;
  }

  virtual bool visit(const luci::CirclePRelu *node) = 0;

  virtual bool visit(const luci::CircleTransposeConv *node) = 0;

  virtual bool visit(const luci::CircleFullyConnected *node) = 0;

  bool visit(const luci::CircleAdd *node)
  {
    // Skip granularity check for indices
    if (node->dtype() == loco::DataType::S32 or node->dtype() == loco::DataType::S64)
      return true;

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

  bool visit(const luci::CircleLocalResponseNormalization *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node))
    RETURN_FALSE_UNLESS(is_lwq(node->input()));
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
    // Skip granularity check for indices
    if (node->dtype() == loco::DataType::S32 or node->dtype() == loco::DataType::S64)
      return true;

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

  bool visit(const luci::CircleOneHot *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node));
    RETURN_FALSE_UNLESS(is_lwq(node->off_value()));
    RETURN_FALSE_UNLESS(is_lwq(node->on_value()));
    return true;
  }

  bool visit(const luci::CircleReduceMax *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node));
    RETURN_FALSE_UNLESS(is_lwq(node->input()));
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
    auto input = loco::must_cast<const luci::CircleNode *>(node->tensor());
    bool input_quantized = input->quantparam() != nullptr;
    bool node_quantized = node->quantparam() != nullptr;
    RETURN_FALSE_UNLESS(input_quantized == node_quantized);
    RETURN_FALSE_UNLESS(not node_quantized or is_lwq(node))
    RETURN_FALSE_UNLESS(not input_quantized or is_lwq(input));
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

  bool visit(const luci::CircleSpaceToBatchND *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node));
    RETURN_FALSE_UNLESS(is_lwq(node->input()));
    return true;
  }

  bool visit(const luci::CircleSpaceToDepth *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node));
    RETURN_FALSE_UNLESS(is_lwq(node->input()));
    return true;
  }

  bool visit(const luci::CircleSlice *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node));
    RETURN_FALSE_UNLESS(is_lwq(node->input()));
    return true;
  }

  bool visit(const luci::CircleSplit *node)
  {
    // node's output is the input of CircleSplitOut, thus not quantized
    RETURN_FALSE_UNLESS(is_lwq(node->input()));
    return true;
  }

  bool visit(const luci::CircleSplitOut *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node));
    return true;
  }

  bool visit(const luci::CircleSplitV *node)
  {
    // node's output is the input of CircleSplitVOut, thus not quantized
    RETURN_FALSE_UNLESS(is_lwq(node->input()));
    return true;
  }

  bool visit(const luci::CircleSplitVOut *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node));
    return true;
  }

  bool visit(const luci::CircleStridedSlice *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node));
    RETURN_FALSE_UNLESS(is_lwq(node->input()));
    return true;
  }

  bool visit(const luci::CircleSum *node)
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

  bool visit(const luci::CircleBatchToSpaceND *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node));
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

  bool visit(const luci::CircleGelu *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node));
    RETURN_FALSE_UNLESS(is_lwq(node->features()));
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

  bool visit(const luci::CircleRsqrt *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node));
    RETURN_FALSE_UNLESS(is_lwq(node->x()));
    return true;
  }

  bool visit(const luci::CircleSqrt *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node));
    RETURN_FALSE_UNLESS(is_lwq(node->x()));
    return true;
  }

  bool visit(const luci::CircleElu *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node));
    RETURN_FALSE_UNLESS(is_lwq(node->features()));
    return true;
  }

  bool visit(const luci::CirclePow *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node));
    RETURN_FALSE_UNLESS(is_lwq(node->x()));
    RETURN_FALSE_UNLESS(is_lwq(node->y()));
    return true;
  }

  bool visit(const luci::CircleResizeBilinear *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node));
    RETURN_FALSE_UNLESS(is_lwq(node->input()));
    return true;
  }

  bool visit(const luci::CircleResizeNearestNeighbor *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node));
    RETURN_FALSE_UNLESS(is_lwq(node->input()));
    return true;
  }

  bool visit(const luci::CircleUnpack *node)
  {
    // node's output is the input of CircleUnpackOut, thus not quantized
    RETURN_FALSE_UNLESS(is_lwq(node->value()));
    return true;
  }

  bool visit(const luci::CircleUnpackOut *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node));
    return true;
  }

  bool visit(const luci::CircleCast *node)
  {
    auto input = loco::must_cast<const luci::CircleNode *>(node->x());
    bool input_quantized = input->quantparam() != nullptr;
    bool node_quantized = node->quantparam() != nullptr;
    RETURN_FALSE_UNLESS(not input_quantized or is_lwq(input));
    RETURN_FALSE_UNLESS(not node_quantized or is_lwq(node));
    return true;
  }

  // TODO: Implement more Ops

  bool visit(const luci::CircleNode *) { return true; }
};

class VerifyQuantizedNodeChannelWiseGranularity final : public VerifyQuantizedNodeGranularity
{
private:
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
    luci::CircleConst *bias = dynamic_cast<luci::CircleConst *>(node->bias());
    if (bias != nullptr)
      RETURN_FALSE_UNLESS(is_cwq_const(node->bias(), rank(node->bias()) - 1))
    return true;
  }

  bool visit(const luci::CircleDepthwiseConv2D *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node))
    RETURN_FALSE_UNLESS(is_lwq(node->input()))
    RETURN_FALSE_UNLESS(is_cwq_const(node->filter(), 3))
    luci::CircleConst *bias = dynamic_cast<luci::CircleConst *>(node->bias());
    if (bias != nullptr)
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
    luci::CircleConst *bias = dynamic_cast<luci::CircleConst *>(node->bias());
    // Bias is optional (it can be CircleOutputExclude)
    if (bias != nullptr)
      RETURN_FALSE_UNLESS(is_cwq_const(node->bias(), rank(node->bias()) - 1))
    return true;
  }
};

class VerifyQuantizedNodeLayerWiseGranularity final : public VerifyQuantizedNodeGranularity
{
private:
  bool is_lwq_const(const loco::Node *node)
  {
    auto circle_node = loco::must_cast<const luci::CircleConst *>(node);

    if (circle_node->quantparam() == nullptr)
      return false;

    if (circle_node->quantparam()->scale.size() != 1)
      return false;

    if (circle_node->quantparam()->zerop.size() != 1)
      return false;

    return true;
  }

private:
  bool visit(const luci::CircleConv2D *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node))
    RETURN_FALSE_UNLESS(is_lwq(node->input()))
    RETURN_FALSE_UNLESS(is_lwq_const(node->filter()))
    luci::CircleConst *bias = dynamic_cast<luci::CircleConst *>(node->bias());
    if (bias != nullptr)
      RETURN_FALSE_UNLESS(is_lwq_const(node->bias()))
    return true;
  }

  bool visit(const luci::CircleDepthwiseConv2D *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node))
    RETURN_FALSE_UNLESS(is_lwq(node->input()))
    RETURN_FALSE_UNLESS(is_lwq_const(node->filter()))
    luci::CircleConst *bias = dynamic_cast<luci::CircleConst *>(node->bias());
    if (bias != nullptr)
      RETURN_FALSE_UNLESS(is_lwq_const(node->bias()))
    return true;
  }

  bool visit(const luci::CircleInstanceNorm *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node))
    RETURN_FALSE_UNLESS(is_lwq(node->input()))
    RETURN_FALSE_UNLESS(is_lwq_const(node->gamma()))
    RETURN_FALSE_UNLESS(is_lwq_const(node->beta()))
    return true;
  }

  bool visit(const luci::CirclePRelu *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node))
    RETURN_FALSE_UNLESS(is_lwq(node->input()))
    RETURN_FALSE_UNLESS(is_lwq_const(node->alpha()))
    return true;
  }

  bool visit(const luci::CircleTransposeConv *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node))
    RETURN_FALSE_UNLESS(is_lwq(node->outBackprop()))
    RETURN_FALSE_UNLESS(is_lwq_const(node->filter()))
    luci::CircleConst *bias = dynamic_cast<luci::CircleConst *>(node->bias());
    if (bias != nullptr)
      RETURN_FALSE_UNLESS(is_lwq_const(node->bias()))
    return true;
  }

  bool visit(const luci::CircleFullyConnected *node)
  {
    RETURN_FALSE_UNLESS(is_lwq(node))
    RETURN_FALSE_UNLESS(is_lwq(node->input()))
    RETURN_FALSE_UNLESS(is_lwq_const(node->weights()))
    luci::CircleConst *bias = dynamic_cast<luci::CircleConst *>(node->bias());
    if (bias != nullptr)
      RETURN_FALSE_UNLESS(is_lwq_const(node->bias()))
    return true;
  }
};

} // namespace luci

#undef RETURN_FALSE_UNLESS

#endif // __LUCI_VERIFY_QUANTIZED_NODE_GRANULARITY_H__
