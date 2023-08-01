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

#ifndef __LUCI_CIRCLE_TYPE_INFERENCE_H__
#define __LUCI_CIRCLE_TYPE_INFERENCE_H__

#include <luci/Service/CircleTypeInferenceRule.h>
#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeVisitor.h>

#include <loco/IR/DataType.h>

namespace luci
{

namespace tinf // namespace for Type Inference
{

struct Rule
{
  bool infer(const luci::CircleNode *, loco::DataType &) const;
};

class Algorithm final : public luci::CircleNodeVisitor<loco::DataType>
{
public:
  // TODO Remove this when all of visit function is implemented
  loco::DataType visit(const luci::CircleNode *node) final
  {
    loco::DataType dtype;
    luci::CircleTypeInferenceRule().infer(node, dtype);
    return dtype;
  }

  // loco::DataType visit(const luci::CircleAbs *node) final;
  // loco::DataType visit(const luci::CircleAdd *node) final;
  // loco::DataType visit(const luci::CircleAddN *node) final;
  // loco::DataType visit(const luci::CircleArgMax *node) final;
  // loco::DataType visit(const luci::CircleArgMin *node) final;
  // loco::DataType visit(const luci::CircleAveragePool2D *node) final;
  // loco::DataType visit(const luci::CircleBatchMatMul *node) final;
  // loco::DataType visit(const luci::CircleBatchToSpaceND *node) final;
  // loco::DataType visit(const luci::CircleCast *node) final;
  // loco::DataType visit(const luci::CircleCeil *node) final;
  // loco::DataType visit(const luci::CircleConcatenation *node) final;
  // loco::DataType visit(const luci::CircleConst *node) final;
  // loco::DataType visit(const luci::CircleConv2D *node) final;
  // loco::DataType visit(const luci::CircleCos *node) final;
  // loco::DataType visit(const luci::CircleCustom *node) final;
  // loco::DataType visit(const luci::CircleDepthToSpace *node) final;
  // loco::DataType visit(const luci::CircleDepthwiseConv2D *node) final;
  // loco::DataType visit(const luci::CircleDequantize *node) final;
  // loco::DataType visit(const luci::CircleDiv *node) final;
  // loco::DataType visit(const luci::CircleElu *node) final;
  // loco::DataType visit(const luci::CircleEqual *node) final;
  // loco::DataType visit(const luci::CircleExp *node) final;
  // loco::DataType visit(const luci::CircleExpandDims *node) final;
  // loco::DataType visit(const luci::CircleFakeQuant *node) final;
  // loco::DataType visit(const luci::CircleFill *node) final;
  // loco::DataType visit(const luci::CircleFloor *node) final;
  // loco::DataType visit(const luci::CircleFloorDiv *node) final;
  // loco::DataType visit(const luci::CircleFloorMod *node) final;
  // loco::DataType visit(const luci::CircleFullyConnected *node) final;
  // loco::DataType visit(const luci::CircleGather *node) final;
  // loco::DataType visit(const luci::CircleGatherNd *node) final;
  // loco::DataType visit(const luci::CircleGreater *node) final;
  // loco::DataType visit(const luci::CircleGreaterEqual *node) final;
  // loco::DataType visit(const luci::CircleHardSwish *node) final;
  // loco::DataType visit(const luci::CircleIf *node) final;
  // loco::DataType visit(const luci::CircleL2Normalize *node) final;
  // loco::DataType visit(const luci::CircleL2Pool2D *node) final;
  // loco::DataType visit(const luci::CircleLeakyRelu *node) final;
  // loco::DataType visit(const luci::CircleLess *node) final;
  // loco::DataType visit(const luci::CircleLessEqual *node) final;
  // loco::DataType visit(const luci::CircleLocalResponseNormalization *node) final;
  // loco::DataType visit(const luci::CircleLog *node) final;
  // loco::DataType visit(const luci::CircleLogicalAnd *node) final;
  // loco::DataType visit(const luci::CircleLogicalNot *node) final;
  // loco::DataType visit(const luci::CircleLogicalOr *node) final;
  // loco::DataType visit(const luci::CircleLogistic *node) final;
  // loco::DataType visit(const luci::CircleLogSoftmax *node) final;
  // loco::DataType visit(const luci::CircleMatrixDiag *node) final;
  // loco::DataType visit(const luci::CircleMatrixSetDiag *node) final;
  // loco::DataType visit(const luci::CircleMaximum *node) final;
  // loco::DataType visit(const luci::CircleMaxPool2D *node) final;
  // loco::DataType visit(const luci::CircleMean *node) final;
  // loco::DataType visit(const luci::CircleMinimum *node) final;
  // loco::DataType visit(const luci::CircleMirrorPad *node) final;
  // loco::DataType visit(const luci::CircleNeg *node) final;
  // loco::DataType visit(const luci::CircleNonMaxSuppressionV4 *node) final;
  // loco::DataType visit(const luci::CircleNonMaxSuppressionV5 *node) final;
  // loco::DataType visit(const luci::CircleNotEqual *node) final;
  // loco::DataType visit(const luci::CirclePack *node) final;
  // loco::DataType visit(const luci::CirclePad *node) final;
  // loco::DataType visit(const luci::CirclePadV2 *node) final;
  // loco::DataType visit(const luci::CirclePow *node) final;
  // loco::DataType visit(const luci::CirclePRelu *node) final;
  // loco::DataType visit(const luci::CircleRange *node) final;
  // loco::DataType visit(const luci::CircleRank *node) final;
  // loco::DataType visit(const luci::CircleMul *node) final;
  // loco::DataType visit(const luci::CircleOneHot *node) final;
  // loco::DataType visit(const luci::CircleQuantize *node) final;
  // loco::DataType visit(const luci::CircleReduceAny *node) final;
  // loco::DataType visit(const luci::CircleReduceMax *node) final;
  // loco::DataType visit(const luci::CircleReduceMin *node) final;
  // loco::DataType visit(const luci::CircleReduceProd *node) final;
  // loco::DataType visit(const luci::CircleRelu *node) final;
  // loco::DataType visit(const luci::CircleRelu6 *node) final;
  // loco::DataType visit(const luci::CircleReluN1To1 *node) final;
  // loco::DataType visit(const luci::CircleReshape *node) final;
  // loco::DataType visit(const luci::CircleResizeBilinear *node) final;
  // loco::DataType visit(const luci::CircleResizeNearestNeighbor *node) final;
  // loco::DataType visit(const luci::CircleReverseSequence *node) final;
  // loco::DataType visit(const luci::CircleReverseV2 *node) final;
  // loco::DataType visit(const luci::CircleRound *node) final;
  // loco::DataType visit(const luci::CircleRsqrt *node) final;
  // loco::DataType visit(const luci::CircleScatterNd *node) final;
  // loco::DataType visit(const luci::CircleSegmentSum *node) final;
  // loco::DataType visit(const luci::CircleSelect *node) final;
  // loco::DataType visit(const luci::CircleSelectV2 *node) final;
  // loco::DataType visit(const luci::CircleShape *node) final;
  // loco::DataType visit(const luci::CircleSin *node) final;
  // loco::DataType visit(const luci::CircleSlice *node) final;
  // loco::DataType visit(const luci::CircleSoftmax *node) final;
  // loco::DataType visit(const luci::CircleSpaceToBatchND *node) final;
  // loco::DataType visit(const luci::CircleSpaceToDepth *node) final;
  // loco::DataType visit(const luci::CircleSparseToDense *node) final;
  // loco::DataType visit(const luci::CircleSplit *node) final;
  // loco::DataType visit(const luci::CircleSplitV *node) final;
  // loco::DataType visit(const luci::CircleSqrt *node) final;
  // loco::DataType visit(const luci::CircleSquare *node) final;
  // loco::DataType visit(const luci::CircleSquaredDifference *node) final;
  // loco::DataType visit(const luci::CircleSqueeze *node) final;
  // loco::DataType visit(const luci::CircleStridedSlice *node) final;
  // loco::DataType visit(const luci::CircleSub *node) final;
  // loco::DataType visit(const luci::CircleSum *node) final;
  // loco::DataType visit(const luci::CircleTanh *node) final;
  // loco::DataType visit(const luci::CircleTile *node) final;
  // loco::DataType visit(const luci::CircleTopKV2 *node) final;
  // loco::DataType visit(const luci::CircleTranspose *node) final;
  // loco::DataType visit(const luci::CircleTransposeConv *node) final;
  // loco::DataType visit(const luci::CircleUnidirectionalSequenceLSTM *node) final;
  // loco::DataType visit(const luci::CircleUnique *node) final;
  // loco::DataType visit(const luci::CircleUnpack *node) final;
  // loco::DataType visit(const luci::CircleWhere *node) final;
  // loco::DataType visit(const luci::CircleWhile *node) final;
  // loco::DataType visit(const luci::CircleZerosLike *node) final;

  // Circle Only
  // loco::DataType visit(const luci::CircleBCQFullyConnected *node) final;
  // loco::DataType visit(const luci::CircleBCQGather *node) final;
  // loco::DataType visit(const luci::CircleInstanceNorm *node) final;

  // Virtual
  // loco::DataType visit(const luci::CircleInput *node) final;
  // loco::DataType visit(const luci::CircleOutput *node) final;
  // loco::DataType visit(const luci::CircleOutputDummy *node) final;
  // loco::DataType visit(const luci::CircleOutputExclude *node) final;
  // loco::DataType visit(const luci::CircleCustomOut *node) final;
  loco::DataType visit(const luci::CircleIfOut *node) final;
  // loco::DataType visit(const luci::CircleNonMaxSuppressionV4Out *node) final;
  // loco::DataType visit(const luci::CircleNonMaxSuppressionV5Out *node) final;
  // loco::DataType visit(const luci::CircleSplitOut *node) final;
  // loco::DataType visit(const luci::CircleSplitVOut *node) final;
  // loco::DataType visit(const luci::CircleTopKV2Out *node) final;
  // loco::DataType visit(const luci::CircleUniqueOut *node) final;
  // loco::DataType visit(const luci::CircleUnpackOut *node) final;
  // loco::DataType visit(const luci::CircleWhileOut *node) final;
};

} // namespace tinf

} // namespace luci

#endif // __LUCI_CIRCLE_TYPE_INFERENCE_H__
