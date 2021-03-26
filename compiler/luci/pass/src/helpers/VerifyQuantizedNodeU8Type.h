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

#ifndef __LUCI_VERIFY_QUANTIZED_NODE_U8_TYPE_H__
#define __LUCI_VERIFY_QUANTIZED_NODE_U8_TYPE_H__

#include "VerifyQuantizedNodeHelper.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeVisitor.h>

// This macro is undef at the end of the file
#define RETURN_FALSE_UNLESS(ARG) \
  if (not(ARG))                  \
  {                              \
    return false;                \
  }

namespace luci
{

namespace verify_quantization
{

struct VerifyQuantizedNodeU8Type final : public luci::CircleNodeVisitor<bool>
{
private:
  bool visit(const luci::CircleConv2D *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::U8))
    RETURN_FALSE_UNLESS(has_type(node->input(), Type::U8))
    RETURN_FALSE_UNLESS(has_type(node->filter(), Type::U8))
    RETURN_FALSE_UNLESS(has_type(node->bias(), Type::S32))
    return true;
  }

  bool visit(const luci::CircleDepthwiseConv2D *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::U8))
    RETURN_FALSE_UNLESS(has_type(node->input(), Type::U8))
    RETURN_FALSE_UNLESS(has_type(node->filter(), Type::U8))
    RETURN_FALSE_UNLESS(has_type(node->bias(), Type::S32))
    return true;
  }

  bool visit(const luci::CircleInstanceNorm *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::U8))
    RETURN_FALSE_UNLESS(has_type(node->input(), Type::U8))
    RETURN_FALSE_UNLESS(has_type(node->gamma(), Type::U8))
    RETURN_FALSE_UNLESS(has_type(node->beta(), Type::U8))
    return true;
  }

  bool visit(const luci::CirclePRelu *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::U8))
    RETURN_FALSE_UNLESS(has_type(node->input(), Type::U8))
    RETURN_FALSE_UNLESS(has_type(node->alpha(), Type::U8))
    return true;
  }

  bool visit(const luci::CircleTransposeConv *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::U8))
    RETURN_FALSE_UNLESS(has_type(node->outBackprop(), Type::U8))
    RETURN_FALSE_UNLESS(has_type(node->filter(), Type::U8))
    luci::CircleConst *bias = dynamic_cast<luci::CircleConst *>(node->bias());
    if (bias != nullptr)
      RETURN_FALSE_UNLESS(has_type(bias, Type::S32))
    return true;
  }

  bool visit(const luci::CircleFullyConnected *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::U8))
    RETURN_FALSE_UNLESS(has_type(node->input(), Type::U8))
    RETURN_FALSE_UNLESS(has_type(node->weights(), Type::U8))
    RETURN_FALSE_UNLESS(has_type(node->bias(), Type::S32))
    return true;
  }

  bool visit(const luci::CircleAdd *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::U8))
    RETURN_FALSE_UNLESS(has_type(node->x(), Type::U8))
    RETURN_FALSE_UNLESS(has_type(node->y(), Type::U8))
    return true;
  }

  bool visit(const luci::CircleAveragePool2D *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::U8))
    RETURN_FALSE_UNLESS(has_type(node->value(), Type::U8))
    return true;
  }

  bool visit(const luci::CircleMaxPool2D *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::U8))
    RETURN_FALSE_UNLESS(has_type(node->value(), Type::U8))
    return true;
  }

  bool visit(const luci::CircleMean *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::U8))
    RETURN_FALSE_UNLESS(has_type(node->input(), Type::U8))
    RETURN_FALSE_UNLESS(has_type(node->reduction_indices(), Type::S32))
    return true;
  }

  bool visit(const luci::CircleMul *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::U8))
    RETURN_FALSE_UNLESS(has_type(node->x(), Type::U8))
    RETURN_FALSE_UNLESS(has_type(node->y(), Type::U8))
    return true;
  }

  bool visit(const luci::CircleRelu *node)
  {
    RETURN_FALSE_UNLESS(has_type(node, Type::U8))
    RETURN_FALSE_UNLESS(has_type(node->features(), Type::U8))
    return true;
  }

  // TODO: Implement more Ops

  bool visit(const luci::CircleNode *) { return true; }
};

} // namespace verify_quantization
} // namespace luci

#undef RETURN_FALSE_UNLESS

#endif // __LUCI_VERIFY_QUNTIZED_NODE_U8_TYPE_H__
