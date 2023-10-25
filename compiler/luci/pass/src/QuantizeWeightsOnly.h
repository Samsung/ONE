/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_QUANTIZE_WEIGHTS_ONLY_H__
#define __LUCI_QUANTIZE_WEIGHTS_ONLY_H__

#include <luci/Pass/QuantizationParameters.h>
#include <luci/IR/CircleNodeVisitor.h>

namespace luci
{

/**
 * @brief QuantizeWeightsOnly quantizes tensors for weights
 * @details Find min/max values on the fly and then quantize
 */
struct QuantizeWeightsOnly final : public luci::CircleNodeMutableVisitor<void>
{
  QuantizeWeightsOnly(loco::DataType input, loco::DataType output, QuantizationGranularity gr)
    : input_type(input), output_type(output), granularity(gr)
  {
  }

  loco::DataType input_type;
  loco::DataType output_type;
  QuantizationGranularity granularity;

private:
  void quantize_weights(luci::CircleConst *weights);

  void visit(luci::CircleConv2D *node);
  void visit(luci::CircleDepthwiseConv2D *node);
  void visit(luci::CircleFullyConnected *node);
  void visit(luci::CircleNode *);
};

} // namespace luci

#endif // __LUCI_QUANTIZE_WEIGHTS_ONLY_H__
