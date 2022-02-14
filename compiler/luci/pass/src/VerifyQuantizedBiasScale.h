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

#ifndef __LUCI_VERIFY_QUANTIZED_BIAS_SCALE_H__
#define __LUCI_VERIFY_QUANTIZED_BIAS_SCALE_H__

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeVisitor.h>

#include <memory>

namespace luci
{

/**
 * @brief Verify the scale of quantized bias node
 * @details
 *
 * Bias of CONV, DCONV, TCONV, FC layers should meet the following condition.
 *
 * bias scale = input scale * weights scale
 */
class VerifyQuantizedBiasScale : public luci::CircleNodeVisitor<bool>
{
public:
  static std::shared_ptr<VerifyQuantizedBiasScale> create()
  {
    return std::make_shared<VerifyQuantizedBiasScale>();
  };

public:
  bool verify(luci::CircleNode *node) { return node->accept(this); }

private:
  // Operators with bias
  bool visit(const luci::CircleConv2D *node);
  bool visit(const luci::CircleDepthwiseConv2D *node);
  bool visit(const luci::CircleFullyConnected *node);
  bool visit(const luci::CircleTransposeConv *node);

  bool visit(const luci::CircleNode *) { return true; }
};

} // namespace luci

#endif // __LUCI_VERIFY_QUANTIZED_BIAS_SCALE_H__
