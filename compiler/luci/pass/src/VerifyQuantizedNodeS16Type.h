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

  bool visit(const luci::CircleRelu *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->features(), Type::S16))
    return true;
  }

  bool visit(const luci::CircleLogistic *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->x(), Type::S16))
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

  bool visit(const luci::CircleArgMax *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, node->output_type()))
    RETURN_FALSE_UNLESS(has_type(node->input(), Type::S16))
    RETURN_FALSE_UNLESS(has_type(node->dimension(), Type::S32) ||
                        has_type(node->dimension(), Type::S64))
    return true;
  }

  // TODO: Implement more Ops

  bool visit(const luci::CircleNode *) { return true; }
};

} // namespace luci

#undef RETURN_FALSE_UNLESS

#endif // __LUCI_VERIFY_QUNTIZED_NODE_S16_TYPE_H__
