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

#ifndef __LUCI_IR_CIRCLENODES_H__
#define __LUCI_IR_CIRCLENODES_H__

#include "Nodes/CircleAbs.h"
#include "Nodes/CircleAdd.h"
#include "Nodes/CircleAddN.h"
#include "Nodes/CircleArgMax.h"
#include "Nodes/CircleArgMin.h"
#include "Nodes/CircleAveragePool2D.h"
#include "Nodes/CircleBatchMatMul.h"
#include "Nodes/CircleBatchToSpaceND.h"
#include "Nodes/CircleBidirectionalSequenceLSTM.h"
#include "Nodes/CircleCast.h"
#include "Nodes/CircleCeil.h"
#include "Nodes/CircleConcatenation.h"
#include "Nodes/CircleConv2D.h"
#include "Nodes/CircleCos.h"
#include "Nodes/CircleCumSum.h"
#include "Nodes/CircleCustom.h"
#include "Nodes/CircleDensify.h"
#include "Nodes/CircleDepthToSpace.h"
#include "Nodes/CircleDepthwiseConv2D.h"
#include "Nodes/CircleDequantize.h"
#include "Nodes/CircleDiv.h"
#include "Nodes/CircleElu.h"
#include "Nodes/CircleEqual.h"
#include "Nodes/CircleExp.h"
#include "Nodes/CircleExpandDims.h"
#include "Nodes/CircleFakeQuant.h"
#include "Nodes/CircleFill.h"
#include "Nodes/CircleFloor.h"
#include "Nodes/CircleFloorDiv.h"
#include "Nodes/CircleFloorMod.h"
#include "Nodes/CircleFullyConnected.h"
#include "Nodes/CircleGather.h"
#include "Nodes/CircleGatherNd.h"
#include "Nodes/CircleGelu.h"
#include "Nodes/CircleGreater.h"
#include "Nodes/CircleGreaterEqual.h"
#include "Nodes/CircleHardSwish.h"
#include "Nodes/CircleIf.h"
#include "Nodes/CircleL2Normalize.h"
#include "Nodes/CircleL2Pool2D.h"
#include "Nodes/CircleLeakyRelu.h"
#include "Nodes/CircleLess.h"
#include "Nodes/CircleLessEqual.h"
#include "Nodes/CircleLocalResponseNormalization.h"
#include "Nodes/CircleLog.h"
#include "Nodes/CircleLogicalAnd.h"
#include "Nodes/CircleLogicalNot.h"
#include "Nodes/CircleLogicalOr.h"
#include "Nodes/CircleLogistic.h"
#include "Nodes/CircleLogSoftmax.h"
#include "Nodes/CircleMatrixDiag.h"
#include "Nodes/CircleMatrixSetDiag.h"
#include "Nodes/CircleMaximum.h"
#include "Nodes/CircleMaxPool2D.h"
#include "Nodes/CircleMean.h"
#include "Nodes/CircleMinimum.h"
#include "Nodes/CircleMirrorPad.h"
#include "Nodes/CircleMul.h"
#include "Nodes/CircleNeg.h"
#include "Nodes/CircleNonMaxSuppressionV4.h"
#include "Nodes/CircleNonMaxSuppressionV5.h"
#include "Nodes/CircleNotEqual.h"
#include "Nodes/CircleOneHot.h"
#include "Nodes/CirclePack.h"
#include "Nodes/CirclePad.h"
#include "Nodes/CirclePadV2.h"
#include "Nodes/CirclePow.h"
#include "Nodes/CircleQuantize.h"
#include "Nodes/CirclePRelu.h"
#include "Nodes/CircleRange.h"
#include "Nodes/CircleRank.h"
#include "Nodes/CircleReduceAny.h"
#include "Nodes/CircleReduceMax.h"
#include "Nodes/CircleReduceMin.h"
#include "Nodes/CircleReduceProd.h"
#include "Nodes/CircleRelu.h"
#include "Nodes/CircleRelu6.h"
#include "Nodes/CircleReluN1To1.h"
#include "Nodes/CircleReshape.h"
#include "Nodes/CircleResizeBilinear.h"
#include "Nodes/CircleResizeNearestNeighbor.h"
#include "Nodes/CircleReverseSequence.h"
#include "Nodes/CircleReverseV2.h"
#include "Nodes/CircleRound.h"
#include "Nodes/CircleRsqrt.h"
#include "Nodes/CircleScatterNd.h"
#include "Nodes/CircleSegmentSum.h"
#include "Nodes/CircleSelect.h"
#include "Nodes/CircleSelectV2.h"
#include "Nodes/CircleShape.h"
#include "Nodes/CircleSin.h"
#include "Nodes/CircleSlice.h"
#include "Nodes/CircleSoftmax.h"
#include "Nodes/CircleSpaceToBatchND.h"
#include "Nodes/CircleSpaceToDepth.h"
#include "Nodes/CircleSparseToDense.h"
#include "Nodes/CircleSplit.h"
#include "Nodes/CircleSplitV.h"
#include "Nodes/CircleSqrt.h"
#include "Nodes/CircleSquare.h"
#include "Nodes/CircleSquaredDifference.h"
#include "Nodes/CircleSqueeze.h"
#include "Nodes/CircleStridedSlice.h"
#include "Nodes/CircleSub.h"
#include "Nodes/CircleSum.h"
#include "Nodes/CircleSVDF.h"
#include "Nodes/CircleTanh.h"
#include "Nodes/CircleTile.h"
#include "Nodes/CircleTopKV2.h"
#include "Nodes/CircleTranspose.h"
#include "Nodes/CircleTransposeConv.h"
#include "Nodes/CircleUnidirectionalSequenceLSTM.h"
#include "Nodes/CircleUnique.h"
#include "Nodes/CircleUnpack.h"
#include "Nodes/CircleWhere.h"
#include "Nodes/CircleWhile.h"
#include "Nodes/CircleZerosLike.h"
// Circle only
#include "Nodes/CircleBCQFullyConnected.h"
#include "Nodes/CircleBCQGather.h"
#include "Nodes/CircleInstanceNorm.h"
// Virtual nodes
#include "Nodes/CircleConst.h"
#include "Nodes/CircleInput.h"
#include "Nodes/CircleOutput.h"
#include "Nodes/CircleVariable.h"
// Multi-output virtual nodes
#include "Nodes/CircleBidirectionalSequenceLSTMOut.h"
#include "Nodes/CircleCustomOut.h"
#include "Nodes/CircleIfOut.h"
#include "Nodes/CircleNonMaxSuppressionV4Out.h"
#include "Nodes/CircleNonMaxSuppressionV5Out.h"
#include "Nodes/CircleSplitOut.h"
#include "Nodes/CircleSplitVOut.h"
#include "Nodes/CircleTopKV2Out.h"
#include "Nodes/CircleUniqueOut.h"
#include "Nodes/CircleUnpackOut.h"
#include "Nodes/CircleWhileOut.h"

#include <loco/IR/Graph.h>

namespace luci
{

/// @brief Link GraphOutput with CircleOutput node
void link(loco::GraphOutput *, CircleOutput *);

/// @brief Link GraphInput with CircleInput node
void link(loco::GraphInput *, CircleInput *);

/// @brief Find a CircleOutput node with a given output index
CircleOutput *output_node(loco::Graph *g, const loco::GraphOutputIndex &index);

/// @brief Find a Pull node with a given input index
CircleInput *input_node(loco::Graph *g, const loco::GraphInputIndex &index);

} // namespace luci

#endif // __LUCI_IR_CIRCLENODES_H__
