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

#ifndef __LUCI_VERIFY_QUANTIZED_NODE_S16_TYPE_H__
#define __LUCI_VERIFY_QUANTIZED_NODE_S16_TYPE_H__

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeVisitor.h>

using Type = loco::DataType;

// This macro is undef at the end of the file
#define RETURN_FALSE_UNLESS(ARG) \
  if (not(ARG))                  \
  {                              \
    return false;                \
  }

namespace luci
{

/**
 * @brief Verify the data type of INT16 quantized node
 * @details
 *
 * Targets to verify
 * - node's output (i.e., node itself)
 * - node's inputs
 */
struct VerifyQuantizedNodeS16Type final : public luci::CircleNodeVisitor<bool>
{
private:
  bool has_type(const loco::Node *node, Type dtype)
  {
    auto circle_node = loco::must_cast<const luci::CircleNode *>(node);
    return circle_node->dtype() == dtype;
  }

private:
  bool visit(const luci::CircleConv2D *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->input(), Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->filter(), Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->bias(), Type::S64))
    return true;
  }

  bool visit(const luci::CircleConcatenation *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::S16))
    for (uint32_t i = 0; i < node->numValues(); i++)
    {
      RETURN_FALSE_UNLESS(has_type(node->values(i), Type::S16))
    }
    return true;
  }

  bool visit(const luci::CircleDepthToSpace *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->input(), Type::S16))
    return true;
  }

  bool visit(const luci::CircleDepthwiseConv2D *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->input(), Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->filter(), Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->bias(), Type::S64))
    return true;
  }

  bool visit(const luci::CircleInstanceNorm *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->input(), Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->gamma(), Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->beta(), Type::S16))
    return true;
  }

  bool visit(const luci::CirclePad *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->input(), Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->paddings(), Type::S32))
    return true;
  }

  bool visit(const luci::CirclePRelu *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->input(), Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->alpha(), Type::S16))
    return true;
  }

  bool visit(const luci::CircleTransposeConv *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->outBackprop(), Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->filter(), Type::S16))
    luci::CircleConst *bias = dynamic_cast<luci::CircleConst *>(node->bias());
    if (bias != nullptr)
      RETURN_FALSE_UNLESS(has_type(bias, Type::S64))
    return true;
  }

  bool visit(const luci::CircleFullyConnected *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->input(), Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->weights(), Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->bias(), Type::S64))
    return true;
  }

  bool visit(const luci::CircleAdd *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->x(), Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->y(), Type::S16))
    return true;
  }

  bool visit(const luci::CircleAveragePool2D *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->value(), Type::S16))
    return true;
  }

  bool visit(const luci::CircleLogicalOr *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::BOOL))
    RETURN_FALSE_UNLESS(has_type(node->x(), Type::BOOL))
    RETURN_FALSE_UNLESS(has_type(node->y(), Type::BOOL))
    return true;
  }

  bool visit(const luci::CircleMaxPool2D *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->value(), Type::S16))
    return true;
  }

  bool visit(const luci::CircleMean *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->input(), Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->reduction_indices(), Type::S32))
    return true;
  }

  bool visit(const luci::CircleMul *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->x(), Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->y(), Type::S16))
    return true;
  }

  bool visit(const luci::CircleNotEqual *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::BOOL))
    RETURN_FALSE_UNLESS(has_type(node->x(), Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->y(), Type::S16))
    return true;
  }

  bool visit(const luci::CircleRelu *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->features(), Type::S16))
    return true;
  }

  bool visit(const luci::CircleReshape *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->tensor(), Type::S16))
    luci::CircleConst *shape = dynamic_cast<luci::CircleConst *>(node->shape());
    if (shape != nullptr)
      RETURN_FALSE_UNLESS(has_type(shape, Type::S32))
    return true;
  }

  bool visit(const luci::CircleLogistic *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->x(), Type::S16))
    return true;
  }

  bool visit(const luci::CircleSoftmax *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->logits(), Type::S16))
    return true;
  }

  bool visit(const luci::CircleSlice *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->input(), Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->begin(), Type::S32) || has_type(node->begin(), Type::S64))
    RETURN_FALSE_UNLESS(has_type(node->size(), Type::S32) || has_type(node->size(), Type::S64))
    return true;
  }

  bool visit(const luci::CircleStridedSlice *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->input(), Type::S16))
    return true;
  }

  bool visit(const luci::CircleArgMax *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, node->output_type()))
    RETURN_FALSE_UNLESS(has_type(node->input(), Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->dimension(), Type::S32) ||
                        has_type(node->dimension(), Type::S64))
    return true;
  }

  bool visit(const luci::CircleBatchToSpaceND *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->input(), Type::S16))
    return true;
  }

  bool visit(const luci::CircleTanh *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->x(), Type::S16))
    return true;
  }

  bool visit(const luci::CircleTranspose *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->a(), Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->perm(), Type::S32))
    return true;
  }

  bool visit(const luci::CircleFloor *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->x(), Type::S16))
    return true;
  }

  bool visit(const luci::CircleGreater *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::BOOL))
    RETURN_FALSE_UNLESS(has_type(node->x(), Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->y(), Type::S16))
    return true;
  }

  bool visit(const luci::CircleGreaterEqual *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::BOOL))
    RETURN_FALSE_UNLESS(has_type(node->x(), Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->y(), Type::S16))
    return true;
  }

  bool visit(const luci::CircleDiv *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->x(), Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->y(), Type::S16))
    return true;
  }

  bool visit(const luci::CircleFloorDiv *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->x(), Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->y(), Type::S16))
    return true;
  }

  bool visit(const luci::CircleRsqrt *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->x(), Type::S16))
    return true;
  }

  bool visit(const luci::CircleSqrt *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->x(), Type::S16))
    return true;
  }

  bool visit(const luci::CircleElu *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->features(), Type::S16))
    return true;
  }

  bool visit(const luci::CirclePow *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->x(), Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->y(), Type::S16))
    return true;
  }

  // TODO: Implement more Ops

  bool visit(const luci::CircleNode *) { return true; }
};

} // namespace luci

#undef RETURN_FALSE_UNLESS

#endif // __LUCI_VERIFY_QUNTIZED_NODE_S16_TYPE_H__
