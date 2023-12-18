/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __LUCI_CIRCLE_SHAPE_INFERENCE_H__
#define __LUCI_CIRCLE_SHAPE_INFERENCE_H__

#include <luci/Service/CircleShapeInferenceRule.h>
#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeVisitor.h>

#include <loco/IR/NodeShape.h>
#include <loco/IR/TensorShape.h>

namespace luci
{

namespace sinf // namespace for Shape Inference
{

struct Rule
{
  bool infer(const luci::CircleNode *, loco::TensorShape &) const;
};

class Algorithm final : public luci::CircleNodeVisitor<loco::TensorShape>
{
public:
  // TODO Remove this when all of visit function is implemented
  loco::TensorShape visit(const luci::CircleNode *node) final
  {
    loco::NodeShape shape;
    luci::CircleShapeInferenceRule().infer(node, shape);
    return shape.as<loco::TensorShape>();
  }

  // loco::TensorShape visit(const luci::CircleAbs *node) final;
  // loco::TensorShape visit(const luci::CircleAdd *node) final;
  // loco::TensorShape visit(const luci::CircleAddN *node) final;
  // loco::TensorShape visit(const luci::CircleArgMax *node) final;
  // loco::TensorShape visit(const luci::CircleArgMin *node) final;
  // loco::TensorShape visit(const luci::CircleAveragePool2D *node) final;
  // loco::TensorShape visit(const luci::CircleBatchMatMul *node) final;
  // loco::TensorShape visit(const luci::CircleBatchToSpaceND *node) final;
  // loco::TensorShape visit(const luci::CircleCast *node) final;
  // loco::TensorShape visit(const luci::CircleCeil *node) final;
  // loco::TensorShape visit(const luci::CircleConcatenation *node) final;
  // loco::TensorShape visit(const luci::CircleConst *node) final;
  // loco::TensorShape visit(const luci::CircleConv2D *node) final;
  // loco::TensorShape visit(const luci::CircleCos *node) final;
  // loco::TensorShape visit(const luci::CircleCustom *node) final;
  // loco::TensorShape visit(const luci::CircleDepthToSpace *node) final;
  // loco::TensorShape visit(const luci::CircleDepthwiseConv2D *node) final;
  // loco::TensorShape visit(const luci::CircleDequantize *node) final;
  // loco::TensorShape visit(const luci::CircleDiv *node) final;
  // loco::TensorShape visit(const luci::CircleElu *node) final;
  // loco::TensorShape visit(const luci::CircleEqual *node) final;
  // loco::TensorShape visit(const luci::CircleExp *node) final;
  // loco::TensorShape visit(const luci::CircleExpandDims *node) final;
  // loco::TensorShape visit(const luci::CircleFakeQuant *node) final;
  // loco::TensorShape visit(const luci::CircleFill *node) final;
  // loco::TensorShape visit(const luci::CircleFloor *node) final;
  // loco::TensorShape visit(const luci::CircleFloorDiv *node) final;
  // loco::TensorShape visit(const luci::CircleFloorMod *node) final;
  // loco::TensorShape visit(const luci::CircleFullyConnected *node) final;
  // loco::TensorShape visit(const luci::CircleGather *node) final;
  // loco::TensorShape visit(const luci::CircleGatherNd *node) final;
  // loco::TensorShape visit(const luci::CircleGreater *node) final;
  // loco::TensorShape visit(const luci::CircleGreaterEqual *node) final;
  // loco::TensorShape visit(const luci::CircleHardSwish *node) final;
  // loco::TensorShape visit(const luci::CircleIf *node) final;
  // loco::TensorShape visit(const luci::CircleL2Normalize *node) final;
  // loco::TensorShape visit(const luci::CircleL2Pool2D *node) final;
  // loco::TensorShape visit(const luci::CircleLeakyRelu *node) final;
  // loco::TensorShape visit(const luci::CircleLess *node) final;
  // loco::TensorShape visit(const luci::CircleLessEqual *node) final;
  // loco::TensorShape visit(const luci::CircleLocalResponseNormalization *node) final;
  // loco::TensorShape visit(const luci::CircleLog *node) final;
  // loco::TensorShape visit(const luci::CircleLogicalAnd *node) final;
  // loco::TensorShape visit(const luci::CircleLogicalNot *node) final;
  // loco::TensorShape visit(const luci::CircleLogicalOr *node) final;
  // loco::TensorShape visit(const luci::CircleLogistic *node) final;
  // loco::TensorShape visit(const luci::CircleLogSoftmax *node) final;
  // loco::TensorShape visit(const luci::CircleMatrixDiag *node) final;
  // loco::TensorShape visit(const luci::CircleMatrixSetDiag *node) final;
  // loco::TensorShape visit(const luci::CircleMaximum *node) final;
  // loco::TensorShape visit(const luci::CircleMaxPool2D *node) final;
  // loco::TensorShape visit(const luci::CircleMean *node) final;
  // loco::TensorShape visit(const luci::CircleMinimum *node) final;
  // loco::TensorShape visit(const luci::CircleMirrorPad *node) final;
  // loco::TensorShape visit(const luci::CircleMul *node) final;
  // loco::TensorShape visit(const luci::CircleNeg *node) final;
  // loco::TensorShape visit(const luci::CircleNonMaxSuppressionV4 *node) final;
  // loco::TensorShape visit(const luci::CircleNonMaxSuppressionV5 *node) final;
  // loco::TensorShape visit(const luci::CircleNotEqual *node) final;
  // loco::TensorShape visit(const luci::CircleOneHot *node) final;
  // loco::TensorShape visit(const luci::CirclePack *node) final;
  // loco::TensorShape visit(const luci::CirclePad *node) final;
  // loco::TensorShape visit(const luci::CirclePadV2 *node) final;
  // loco::TensorShape visit(const luci::CirclePow *node) final;
  // loco::TensorShape visit(const luci::CirclePRelu *node) final;
  // loco::TensorShape visit(const luci::CircleQuantize *node) final;
  // loco::TensorShape visit(const luci::CircleRange *node) final;
  // loco::TensorShape visit(const luci::CircleRank *node) final;
  // loco::TensorShape visit(const luci::CircleReduceAny *node) final;
  // loco::TensorShape visit(const luci::CircleReduceMax *node) final;
  // loco::TensorShape visit(const luci::CircleReduceMin *node) final;
  // loco::TensorShape visit(const luci::CircleReduceProd *node) final;
  // loco::TensorShape visit(const luci::CircleRelu *node) final;
  // loco::TensorShape visit(const luci::CircleRelu6 *node) final;
  // loco::TensorShape visit(const luci::CircleReluN1To1 *node) final;
  // loco::TensorShape visit(const luci::CircleReshape *node) final;
  // loco::TensorShape visit(const luci::CircleResizeBilinear *node) final;
  // loco::TensorShape visit(const luci::CircleResizeNearestNeighbor *node) final;
  // loco::TensorShape visit(const luci::CircleReverseSequence *node) final;
  // loco::TensorShape visit(const luci::CircleReverseV2 *node) final;
  // loco::TensorShape visit(const luci::CircleRound *node) final;
  // loco::TensorShape visit(const luci::CircleRsqrt *node) final;
  // loco::TensorShape visit(const luci::CircleScatterNd *node) final;
  // loco::TensorShape visit(const luci::CircleSegmentSum *node) final;
  // loco::TensorShape visit(const luci::CircleSelect *node) final;
  // loco::TensorShape visit(const luci::CircleSelectV2 *node) final;
  // loco::TensorShape visit(const luci::CircleShape *node) final;
  // loco::TensorShape visit(const luci::CircleSin *node) final;
  // loco::TensorShape visit(const luci::CircleSlice *node) final;
  // loco::TensorShape visit(const luci::CircleSoftmax *node) final;
  // loco::TensorShape visit(const luci::CircleSpaceToBatchND *node) final;
  // loco::TensorShape visit(const luci::CircleSpaceToDepth *node) final;
  // loco::TensorShape visit(const luci::CircleSparseToDense *node) final;
  // loco::TensorShape visit(const luci::CircleSplit *node) final;
  // loco::TensorShape visit(const luci::CircleSplitV *node) final;
  // loco::TensorShape visit(const luci::CircleSqrt *node) final;
  // loco::TensorShape visit(const luci::CircleSquare *node) final;
  // loco::TensorShape visit(const luci::CircleSquaredDifference *node) final;
  // loco::TensorShape visit(const luci::CircleSqueeze *node) final;
  // loco::TensorShape visit(const luci::CircleStridedSlice *node) final;
  // loco::TensorShape visit(const luci::CircleSub *node) final;
  // loco::TensorShape visit(const luci::CircleSum *node) final;
  // loco::TensorShape visit(const luci::CircleTanh *node) final;
  // loco::TensorShape visit(const luci::CircleTile *node) final;
  // loco::TensorShape visit(const luci::CircleTopKV2 *node) final;
  // loco::TensorShape visit(const luci::CircleTranspose *node) final;
  // loco::TensorShape visit(const luci::CircleTransposeConv *node) final;
  // loco::TensorShape visit(const luci::CircleUnidirectionalSequenceLSTM *node) final;
  // loco::TensorShape visit(const luci::CircleUnique *node) final;
  // loco::TensorShape visit(const luci::CircleUnpack *node) final;
  // loco::TensorShape visit(const luci::CircleWhere *node) final;
  // loco::TensorShape visit(const luci::CircleWhile *node) final;
  // loco::TensorShape visit(const luci::CircleZerosLike *node) final;

  // Circle Only
  // loco::TensorShape visit(const luci::CircleBCQFullyConnected *node) final;
  // loco::TensorShape visit(const luci::CircleBCQGather *node) final;
  // loco::TensorShape visit(const luci::CircleInstanceNorm *node) final;
  // loco::TensorShape visit(const luci::CircleGRU *node) final;

  // Virtual
  // loco::TensorShape visit(const luci::CircleCustomOut *node) final;
  loco::TensorShape visit(const luci::CircleIfOut *node) final;
  // loco::TensorShape visit(const luci::CircleInput *node) final;
  // loco::TensorShape visit(const luci::CircleNonMaxSuppressionV4Out *node) final;
  // loco::TensorShape visit(const luci::CircleNonMaxSuppressionV5Out *node) final;
  // loco::TensorShape visit(const luci::CircleOutput *node) final;
  // loco::TensorShape visit(const luci::CircleOutputDummy *node) final;
  // loco::TensorShape visit(const luci::CircleOutputExclude *node) final;
  // loco::TensorShape visit(const luci::CircleSplitOut *node) final;
  // loco::TensorShape visit(const luci::CircleSplitVOut *node) final;
  // loco::TensorShape visit(const luci::CircleTopKV2Out *node) final;
  // loco::TensorShape visit(const luci::CircleUniqueOut *node) final;
  // loco::TensorShape visit(const luci::CircleUnpackOut *node) final;
  // loco::TensorShape visit(const luci::CircleWhileOut *node) final;
};

} // namespace sinf

} // namespace luci

#endif // __LUCI_CIRCLE_SHAPE_INFERENCE_H__
