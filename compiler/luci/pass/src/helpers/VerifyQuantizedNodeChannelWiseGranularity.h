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

struct VerifyQuantizedNodeChannelWiseGranularity final : public luci::CircleNodeVisitor<bool>
{
private:
  bool visit(const luci::CircleConv2D *node)
  {
    RETURN_FALSE_UNLESS(is_cwq_const(node->filter(), 0))
    RETURN_FALSE_UNLESS(is_cwq_const(node->bias(), rank(node->bias()) - 1))
    return true;
  }

  bool visit(const luci::CircleDepthwiseConv2D *node)
  {
    RETURN_FALSE_UNLESS(is_cwq_const(node->filter(), 3))
    RETURN_FALSE_UNLESS(is_cwq_const(node->bias(), rank(node->bias()) - 1))
    return true;
  }

  bool visit(const luci::CircleInstanceNorm *node)
  {
    RETURN_FALSE_UNLESS(is_cwq_const(node->gamma(), rank(node->gamma()) - 1))
    RETURN_FALSE_UNLESS(is_cwq_const(node->beta(), rank(node->beta()) - 1))
    return true;
  }

  bool visit(const luci::CirclePRelu *node)
  {
    RETURN_FALSE_UNLESS(is_cwq_const(node->alpha(), rank(node->alpha()) - 1))
    return true;
  }

  bool visit(const luci::CircleTransposeConv *node)
  {
    RETURN_FALSE_UNLESS(is_cwq_const(node->filter(), 0))
    luci::CircleConst *bias = dynamic_cast<luci::CircleConst *>(node->bias());
    if (bias != nullptr)
      RETURN_FALSE_UNLESS(is_cwq_const(node->bias(), rank(node->bias()) - 1))

    return true;
  }

  bool visit(const luci::CircleFullyConnected *node)
  {
    RETURN_FALSE_UNLESS(is_cwq_const(node->weights(), 0))
    RETURN_FALSE_UNLESS(is_cwq_const(node->bias(), rank(node->bias()) - 1))
    return true;
  }

  // These operators do not have CWQ constants
  bool visit(const luci::CircleAdd *) { return true; }
  bool visit(const luci::CircleAveragePool2D *) { return true; }
  bool visit(const luci::CircleMaxPool2D *) { return true; }
  bool visit(const luci::CircleMean *) { return true; }
  bool visit(const luci::CircleMul *) { return true; }
  bool visit(const luci::CircleRelu *) { return true; }

  // TODO: Implement more Ops

  bool visit(const luci::CircleNode *) { return true; }
};

} // namespace verify_quantization
} // namespace luci

#undef RETURN_FALSE_UNLESS

#endif // __LUCI_VERIFY_QUANTIZED_NODE_CHANNELWISE_GRANULARITY_H__
