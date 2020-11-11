/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_CIRCLE_SHAPE_SIGNATURE_INFERENCE_RULE_H__
#define __LUCI_CIRCLE_SHAPE_SIGNATURE_INFERENCE_RULE_H__

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeVisitor.h>
#include <luci/IR/CircleShapeSignature.h>

namespace luci
{

struct CircleShapeSignatureInferenceRule
{
  bool infer(const luci::CircleNode *, ShapeSignature &) const;
};

class ShapeSignatureInferenceAlgorithm final : public luci::CircleNodeVisitor<ShapeSignature>
{
public:
  // TODO Remove this when visit function is implemented for all the operations.
  ShapeSignature visit(const luci::CircleNode *node) final { return node->shape_signature(); }

  // ShapeSignature visit(const luci::CircleAbs *node) final;
  // ShapeSignature visit(const luci::CircleAdd *node) final;
  // ShapeSignature visit(const luci::CircleAddN *node) final;
  // ShapeSignature visit(const luci::CircleArgMax *node) final;
  // ShapeSignature visit(const luci::CircleArgMin *node) final;
  // ShapeSignature visit(const luci::CircleAveragePool2D *node) final;
  // ShapeSignature visit(const luci::CircleBatchMatMul *node) final;
  // ShapeSignature visit(const luci::CircleBatchToSpaceND *node) final;
  // ShapeSignature visit(const luci::CircleCast *node) final;
  // ShapeSignature visit(const luci::CircleCeil *node) final;
  // ShapeSignature visit(const luci::CircleConcatenation *node) final;
  // ShapeSignature visit(const luci::CircleConst *node) final;
  // ShapeSignature visit(const luci::CircleConv2D *node) final;
  // ShapeSignature visit(const luci::CircleCos *node) final;
  // ShapeSignature visit(const luci::CircleCustom *node) final;
  // ShapeSignature visit(const luci::CircleDepthToSpace *node) final;
  // ShapeSignature visit(const luci::CircleDepthwiseConv2D *node) final;
  // ShapeSignature visit(const luci::CircleDequantize *node) final;
  // ShapeSignature visit(const luci::CircleDiv *node) final;
  // ShapeSignature visit(const luci::CircleElu *node) final;
  // ShapeSignature visit(const luci::CircleEqual *node) final;
  // ShapeSignature visit(const luci::CircleExp *node) final;
  // ShapeSignature visit(const luci::CircleExpandDims *node) final;
  // ShapeSignature visit(const luci::CircleFill *node) final;
  // ShapeSignature visit(const luci::CircleFloor *node) final;
  // ShapeSignature visit(const luci::CircleFloorDiv *node) final;
  // ShapeSignature visit(const luci::CircleFloorMod *node) final;
  // ShapeSignature visit(const luci::CircleFullyConnected *node) final;
  // ShapeSignature visit(const luci::CircleGather *node) final;
  // ShapeSignature visit(const luci::CircleGatherNd *node) final;
  // ShapeSignature visit(const luci::CircleGreater *node) final;
  // ShapeSignature visit(const luci::CircleGreaterEqual *node) final;
  // ShapeSignature visit(const luci::CircleIf *node) final;
  // ShapeSignature visit(const luci::CircleL2Normalize *node) final;
  // ShapeSignature visit(const luci::CircleL2Pool2D *node) final;
  // ShapeSignature visit(const luci::CircleLeakyRelu *node) final;
  // ShapeSignature visit(const luci::CircleLess *node) final;
  // ShapeSignature visit(const luci::CircleLessEqual *node) final;
  // ShapeSignature visit(const luci::CircleLocalResponseNormalization *node) final;
  // ShapeSignature visit(const luci::CircleLog *node) final;
  // ShapeSignature visit(const luci::CircleLogicalAnd *node) final;
  // ShapeSignature visit(const luci::CircleLogicalNot *node) final;
  // ShapeSignature visit(const luci::CircleLogicalOr *node) final;
  // ShapeSignature visit(const luci::CircleLogistic *node) final;
  // ShapeSignature visit(const luci::CircleLogSoftmax *node) final;
  // ShapeSignature visit(const luci::CircleMatrixDiag *node) final;
  // ShapeSignature visit(const luci::CircleMatrixSetDiag *node) final;
  // ShapeSignature visit(const luci::CircleMaximum *node) final;
  // ShapeSignature visit(const luci::CircleMaxPool2D *node) final;
  // ShapeSignature visit(const luci::CircleMean *node) final;
  // ShapeSignature visit(const luci::CircleMinimum *node) final;
  // ShapeSignature visit(const luci::CircleMirrorPad *node) final;
  // ShapeSignature visit(const luci::CircleNeg *node) final;
  // ShapeSignature visit(const luci::CircleNonMaxSuppressionV4 *node) final;
  // ShapeSignature visit(const luci::CircleNonMaxSuppressionV5 *node) final;
  // ShapeSignature visit(const luci::CircleNotEqual *node) final;
  // ShapeSignature visit(const luci::CirclePack *node) final;
  // ShapeSignature visit(const luci::CirclePad *node) final;
  // ShapeSignature visit(const luci::CirclePadV2 *node) final;
  // ShapeSignature visit(const luci::CirclePow *node) final;
  // ShapeSignature visit(const luci::CirclePRelu *node) final;
  // ShapeSignature visit(const luci::CircleRange *node) final;
  // ShapeSignature visit(const luci::CircleRank *node) final;
  // ShapeSignature visit(const luci::CircleMul *node) final;
  // ShapeSignature visit(const luci::CircleOneHot *node) final;
  // ShapeSignature visit(const luci::CircleReduceAny *node) final;
  // ShapeSignature visit(const luci::CircleReduceMax *node) final;
  // ShapeSignature visit(const luci::CircleReduceMin *node) final;
  // ShapeSignature visit(const luci::CircleReduceProd *node) final;
  // ShapeSignature visit(const luci::CircleRelu *node) final;
  // ShapeSignature visit(const luci::CircleRelu6 *node) final;
  // ShapeSignature visit(const luci::CircleReluN1To1 *node) final;
  // ShapeSignature visit(const luci::CircleReshape *node) final;
  // ShapeSignature visit(const luci::CircleResizeBilinear *node) final;
  // ShapeSignature visit(const luci::CircleResizeNearestNeighbor *node) final;
  // ShapeSignature visit(const luci::CircleReverseSequence *node) final;
  // ShapeSignature visit(const luci::CircleReverseV2 *node) final;
  // ShapeSignature visit(const luci::CircleRound *node) final;
  // ShapeSignature visit(const luci::CircleRsqrt *node) final;
  // ShapeSignature visit(const luci::CircleScatterNd *node) final;
  // ShapeSignature visit(const luci::CircleSegmentSum *node) final;
  // ShapeSignature visit(const luci::CircleSelect *node) final;
  // ShapeSignature visit(const luci::CircleSelectV2 *node) final;
  // ShapeSignature visit(const luci::CircleShape *node) final;
  // ShapeSignature visit(const luci::CircleSin *node) final;
  // ShapeSignature visit(const luci::CircleSlice *node) final;
  // ShapeSignature visit(const luci::CircleSoftmax *node) final;
  // ShapeSignature visit(const luci::CircleSpaceToBatchND *node) final;
  // ShapeSignature visit(const luci::CircleSpaceToDepth *node) final;
  // ShapeSignature visit(const luci::CircleSparseToDense *node) final;
  // ShapeSignature visit(const luci::CircleSplit *node) final;
  // ShapeSignature visit(const luci::CircleSplitV *node) final;
  // ShapeSignature visit(const luci::CircleSqrt *node) final;
  // ShapeSignature visit(const luci::CircleSquare *node) final;
  // ShapeSignature visit(const luci::CircleSquaredDifference *node) final;
  // ShapeSignature visit(const luci::CircleSqueeze *node) final;
  // ShapeSignature visit(const luci::CircleStridedSlice *node) final;
  // ShapeSignature visit(const luci::CircleSub *node) final;
  // ShapeSignature visit(const luci::CircleSum *node) final;
  // ShapeSignature visit(const luci::CircleTanh *node) final;
  // ShapeSignature visit(const luci::CircleTile *node) final;
  // ShapeSignature visit(const luci::CircleTopKV2 *node) final;
  // ShapeSignature visit(const luci::CircleTranspose *node) final;
  // ShapeSignature visit(const luci::CircleTransposeConv *node) final;
  // ShapeSignature visit(const luci::CircleUnidirectionalSequenceLSTM *node) final;
  // ShapeSignature visit(const luci::CircleUnique *node) final;
  // ShapeSignature visit(const luci::CircleUnpack *node) final;
  // ShapeSignature visit(const luci::CircleWhere *node) final ;
  // ShapeSignature visit(const luci::CircleWhile *node) final;
  // ShapeSignature visit(const luci::CircleZerosLike *node) final;

  // Circle Only
  // ShapeSignature visit(const luci::CircleBCQFullyConnected *node) final;
  // ShapeSignature visit(const luci::CircleBCQGather *node) final;
  // ShapeSignature visit(const luci::CircleInstanceNorm *node) final;

  // Virtual
  // ShapeSignature visit(const luci::CircleInput *node) final;
  // ShapeSignature visit(const luci::CircleOutput *node) final;
  // ShapeSignature visit(const luci::CircleOutputDummy *node) final;
  // ShapeSignature visit(const luci::CircleOutputExclude *node) final;
  // ShapeSignature visit(const luci::CircleCustomOut *node) final;
  // ShapeSignature visit(const luci::CircleIfOut *node) final;
  // ShapeSignature visit(const luci::CircleNonMaxSuppressionV4Out *node) final;
  // ShapeSignature visit(const luci::CircleNonMaxSuppressionV5Out *node) final;
  // ShapeSignature visit(const luci::CircleSplitOut *node) final;
  // ShapeSignature visit(const luci::CircleSplitVOut *node) final;
  // ShapeSignature visit(const luci::CircleTopKV2Out *node) final;
  // ShapeSignature visit(const luci::CircleUniqueOut *node) final;
  // ShapeSignature visit(const luci::CircleUnpackOut *node) final;
  // ShapeSignature visit(const luci::CircleWhileOut *node) final;
};

} // namespace luci

#endif // __LUCI_CIRCLE_SHAPE_SIGNATURE_INFERENCE_RULE_H__
