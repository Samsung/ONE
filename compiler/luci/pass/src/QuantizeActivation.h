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

#ifndef __LUCI_QUANTIZATION_ACTIVATION_H__
#define __LUCI_QUANTIZATION_ACTIVATION_H__

#include <luci/IR/CircleNodeVisitor.h>

namespace luci
{

/**
 * @brief Quantize non-const activation using recorded min/max values
 */
struct QuantizeActivation final : public luci::CircleNodeMutableVisitor<void>
{
  QuantizeActivation(loco::DataType output) : output_type(output) {}

  loco::DataType output_type;

  // Quantize each node using recorded min/max
  void visit(luci::CircleNode *node);
};

/**
 * @brief Quantize non-const activaion using pre-defined scale/zp for special Ops
 */
struct QuantizeSpecialActivation final : public luci::CircleNodeMutableVisitor<void>
{
  QuantizeSpecialActivation(loco::DataType output) : output_type(output) {}

  loco::DataType output_type;

  void visit(luci::CircleNode *node);
  void visit(luci::CircleLogistic *node);
  void visit(luci::CircleTanh *node);
  void visit(luci::CircleSoftmax *node);
  void visit(luci::CircleFloor *node);
  void visit(luci::CircleFloorDiv *node);
  void visit(luci::CircleFloorMod *node);
  void visit(luci::CircleCeil *node);
};

// Quantize constant input activation of a node
// The input of a node is quantized if it is
// 1. Constant (instance of CircleConst*)
// 2. Activation (other inputs e.g., weights, bias, axis, etc should not be quantized here)
struct QuantizeConstInputActivation final : public luci::CircleNodeMutableVisitor<void>
{
  QuantizeConstInputActivation(loco::DataType output_type) : _output_type(output_type) {}

private:
  loco::DataType _output_type;

// Skip NODE
#define SKIP(NODE) \
  void visit(NODE *) {}

  // Handled in QuantizeWeights and QuantizeBias
  SKIP(luci::CircleConv2D)
  SKIP(luci::CircleDepthwiseConv2D)
  SKIP(luci::CircleFullyConnected)
  SKIP(luci::CircleInstanceNorm)
  SKIP(luci::CirclePRelu)
  SKIP(luci::CircleTransposeConv)

  // Handled in PropagateQParamBackwardPass
  SKIP(luci::CircleConcatenation)
  SKIP(luci::CirclePadV2)
  SKIP(luci::CirclePack)
  SKIP(luci::CircleOneHot)

  // Inputs of logical Ops are bool, thus not quantized
  SKIP(luci::CircleLogicalOr)
  SKIP(luci::CircleLogicalAnd)
  SKIP(luci::CircleLogicalNot)

#undef SKIP

  // Default behavior (NYI)
  void visit(luci::CircleNode *node);

  // Ops that receive a single activation as an input
  void visit(luci::CircleAbs *node);
  void visit(luci::CircleArgMax *node);
  void visit(luci::CircleArgMin *node);
  void visit(luci::CircleBatchToSpaceND *node);
  void visit(luci::CircleDepthToSpace *node);
  void visit(luci::CircleElu *node);
  void visit(luci::CircleExp *node);
  void visit(luci::CircleFloor *node);
  void visit(luci::CircleGather *node);
  void visit(luci::CircleGelu *node);
  void visit(luci::CircleLocalResponseNormalization *node);
  void visit(luci::CircleLogistic *node);
  void visit(luci::CircleMean *node);
  void visit(luci::CircleMirrorPad *node);
  void visit(luci::CirclePad *node);
  void visit(luci::CircleReduceAny *node);
  void visit(luci::CircleReduceProd *node);
  void visit(luci::CircleReduceMax *node);
  void visit(luci::CircleReduceMin *node);
  void visit(luci::CircleReshape *node);
  void visit(luci::CircleResizeBilinear *node);
  void visit(luci::CircleResizeNearestNeighbor *node);
  void visit(luci::CircleReverseSequence *node);
  void visit(luci::CircleRsqrt *node);
  void visit(luci::CircleSlice *node);
  void visit(luci::CircleSoftmax *node);
  void visit(luci::CircleSpaceToBatchND *node);
  void visit(luci::CircleSpaceToDepth *node);
  void visit(luci::CircleSplit *node);
  void visit(luci::CircleSplitV *node);
  void visit(luci::CircleSqrt *node);
  void visit(luci::CircleSqueeze *node);
  void visit(luci::CircleStridedSlice *node);
  void visit(luci::CircleSum *node);
  void visit(luci::CircleTanh *node);
  void visit(luci::CircleTile *node);
  void visit(luci::CircleTopKV2 *node);
  void visit(luci::CircleTranspose *node);
  void visit(luci::CircleUnpack *node);

  // Ops that receive two activations as inputs
  void visit(luci::CircleAdd *node);
  void visit(luci::CircleBatchMatMul *node);
  void visit(luci::CircleDiv *node);
  void visit(luci::CircleEqual *node);
  void visit(luci::CircleFloorDiv *node);
  void visit(luci::CircleFloorMod *node);
  void visit(luci::CircleGreater *node);
  void visit(luci::CircleGreaterEqual *node);
  void visit(luci::CircleLess *node);
  void visit(luci::CircleLessEqual *node);
  void visit(luci::CircleMaximum *node);
  void visit(luci::CircleMinimum *node);
  void visit(luci::CircleMul *node);
  void visit(luci::CircleNotEqual *node);
  void visit(luci::CirclePow *node);
  void visit(luci::CircleSelectV2 *node);
  void visit(luci::CircleSub *node);

  // AddN has arbitrary number of inputs
  void visit(luci::CircleAddN *node);
};

} // namespace luci

#endif // __LUCI_QUANTIZATION_ACTIVATION_H__
