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

#ifndef __LUCI_QUANTIZE_BIAS_H__
#define __LUCI_QUANTIZE_BIAS_H__

#include <luci/Pass/QuantizationParameters.h>
#include <luci/IR/CircleNodeVisitor.h>

namespace luci
{

/**
 * @brief QuantizeBias quantizes tensors for bias
 * @details Use input/weights scale to quantize values
 */
struct QuantizeBias final : public luci::CircleNodeMutableVisitor<void>
{
  QuantizeBias(loco::DataType input, loco::DataType output, QuantizationGranularity gr)
    : input_type(input), output_type(output), granularity(gr)
  {
  }

  loco::DataType input_type;
  loco::DataType output_type;
  QuantizationGranularity granularity;

private:
  // Return a quantized bias node
  CircleConst *quantized_bias(CircleNode *input, const CircleNode *weight, CircleNode *bias);

  void visit(luci::CircleConv2D *node);
  void visit(luci::CircleDepthwiseConv2D *node);
  void visit(luci::CircleTransposeConv *node);
  void visit(luci::CircleFullyConnected *node);

  // Default behavior
  void visit(luci::CircleNode *) {}
};

} // namespace luci

#endif // __LUCI_QUANTIZE_BIAS_H__
