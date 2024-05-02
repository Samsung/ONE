/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_PARTITION_CONNECT_NODE_H__
#define __LUCI_PARTITION_CONNECT_NODE_H__

#include <luci/IR/CircleNode.h>
#include <luci/IR/CircleNodeVisitor.h>

namespace luci
{

/**
 * @note MapNode2Clone is used as a map from original node to cloned node
 *       to find input of a cloned node
 *
 *   (Original)              (Clone)
 *
 *     [A]                  [A']
 *      |   [B]              |   [B']
 *      |    |               |    |
 *       \  /                 \  /
 *        [C]                 [C']
 *
 *  From view of [C'] we need to find [A'] and [B']. We know [C] from [C'],
 *  then we can get from input of [C] as [A], [B] then [A]->[A'] and [B]->[B']
 *  from the map.
 */
using MapNode2Clone = std::map<const CircleNode * /* ORG */, CircleNode * /* CLONE */>;

struct CloneContext
{
  std::pair<MapNode2Clone::iterator, bool> emplace(const CircleNode *org, CircleNode *clone)
  {
    return node2clone.emplace(org, clone);
  }
  MapNode2Clone::iterator find(const CircleNode *org) { return node2clone.find(org); }
  MapNode2Clone::iterator end(void) { return node2clone.end(); }

  MapNode2Clone::const_iterator find(const CircleNode *org) const { return node2clone.find(org); }
  MapNode2Clone::const_iterator end(void) const { return node2clone.end(); }

  MapNode2Clone node2clone;
};

class ConnectNode final : public luci::CircleNodeVisitor<void>
{
public:
  ConnectNode(luci::CloneContext &clonecontext) : _clonecontext(clonecontext){};

public:
  void visit(const luci::CircleAbs *) final;
  void visit(const luci::CircleAdd *) final;
  void visit(const luci::CircleAddN *) final;
  void visit(const luci::CircleArgMax *) final;
  void visit(const luci::CircleArgMin *) final;
  void visit(const luci::CircleAveragePool2D *) final;
  void visit(const luci::CircleBatchMatMul *) final;
  void visit(const luci::CircleBatchToSpaceND *) final;
  void visit(const luci::CircleBroadcastTo *) final;
  void visit(const luci::CircleCast *) final;
  void visit(const luci::CircleCeil *) final;
  void visit(const luci::CircleConcatenation *) final;
  void visit(const luci::CircleConst *) final;
  void visit(const luci::CircleConv2D *) final;
  void visit(const luci::CircleCos *) final;
  void visit(const luci::CircleCumSum *) final;
  void visit(const luci::CircleCustom *) final;
  void visit(const luci::CircleDensify *) final;
  void visit(const luci::CircleDepthToSpace *) final;
  void visit(const luci::CircleDepthwiseConv2D *) final;
  void visit(const luci::CircleDequantize *) final;
  void visit(const luci::CircleDiv *) final;
  void visit(const luci::CircleElu *) final;
  void visit(const luci::CircleEqual *) final;
  void visit(const luci::CircleExp *) final;
  void visit(const luci::CircleExpandDims *) final;
  void visit(const luci::CircleFakeQuant *) final;
  void visit(const luci::CircleFill *) final;
  void visit(const luci::CircleFloor *) final;
  void visit(const luci::CircleFloorDiv *) final;
  void visit(const luci::CircleFloorMod *) final;
  void visit(const luci::CircleFullyConnected *) final;
  void visit(const luci::CircleGather *) final;
  void visit(const luci::CircleGatherNd *) final;
  void visit(const luci::CircleGelu *) final;
  void visit(const luci::CircleGreater *) final;
  void visit(const luci::CircleGreaterEqual *) final;
  void visit(const luci::CircleHardSwish *) final;
  void visit(const luci::CircleIf *) final;
  void visit(const luci::CircleL2Normalize *) final;
  void visit(const luci::CircleL2Pool2D *) final;
  void visit(const luci::CircleLeakyRelu *) final;
  void visit(const luci::CircleLess *) final;
  void visit(const luci::CircleLessEqual *) final;
  void visit(const luci::CircleLocalResponseNormalization *) final;
  void visit(const luci::CircleLog *) final;
  void visit(const luci::CircleLogicalAnd *) final;
  void visit(const luci::CircleLogicalNot *) final;
  void visit(const luci::CircleLogicalOr *) final;
  void visit(const luci::CircleLogistic *) final;
  void visit(const luci::CircleLogSoftmax *) final;
  void visit(const luci::CircleMatrixDiag *) final;
  void visit(const luci::CircleMatrixSetDiag *) final;
  void visit(const luci::CircleMaximum *) final;
  void visit(const luci::CircleMaxPool2D *) final;
  void visit(const luci::CircleMean *) final;
  void visit(const luci::CircleMinimum *) final;
  void visit(const luci::CircleMirrorPad *) final;
  void visit(const luci::CircleMul *) final;
  void visit(const luci::CircleNeg *) final;
  void visit(const luci::CircleNonMaxSuppressionV4 *) final;
  void visit(const luci::CircleNonMaxSuppressionV5 *) final;
  void visit(const luci::CircleNotEqual *) final;
  void visit(const luci::CircleOneHot *) final;
  void visit(const luci::CirclePack *) final;
  void visit(const luci::CirclePad *) final;
  void visit(const luci::CirclePadV2 *) final;
  void visit(const luci::CirclePow *) final;
  void visit(const luci::CirclePRelu *) final;
  void visit(const luci::CircleQuantize *) final;
  void visit(const luci::CircleRange *) final;
  void visit(const luci::CircleRank *) final;
  void visit(const luci::CircleReduceAny *) final;
  void visit(const luci::CircleReduceMax *) final;
  void visit(const luci::CircleReduceMin *) final;
  void visit(const luci::CircleReduceProd *) final;
  void visit(const luci::CircleRelu *) final;
  void visit(const luci::CircleRelu0To1 *) final;
  void visit(const luci::CircleRelu6 *) final;
  void visit(const luci::CircleReluN1To1 *) final;
  void visit(const luci::CircleReshape *) final;
  void visit(const luci::CircleResizeBilinear *) final;
  void visit(const luci::CircleResizeNearestNeighbor *) final;
  void visit(const luci::CircleReverseSequence *) final;
  void visit(const luci::CircleReverseV2 *) final;
  void visit(const luci::CircleRound *) final;
  void visit(const luci::CircleRsqrt *) final;
  void visit(const luci::CircleScatterNd *) final;
  void visit(const luci::CircleSegmentSum *) final;
  void visit(const luci::CircleSelect *) final;
  void visit(const luci::CircleSelectV2 *) final;
  void visit(const luci::CircleShape *) final;
  void visit(const luci::CircleSin *) final;
  void visit(const luci::CircleSlice *) final;
  void visit(const luci::CircleSoftmax *) final;
  void visit(const luci::CircleSpaceToBatchND *) final;
  void visit(const luci::CircleSpaceToDepth *) final;
  void visit(const luci::CircleSparseToDense *) final;
  void visit(const luci::CircleSplit *) final;
  void visit(const luci::CircleSplitV *) final;
  void visit(const luci::CircleSqrt *) final;
  void visit(const luci::CircleSquare *) final;
  void visit(const luci::CircleSquaredDifference *) final;
  void visit(const luci::CircleSqueeze *) final;
  void visit(const luci::CircleStridedSlice *) final;
  void visit(const luci::CircleSVDF *) final;
  void visit(const luci::CircleSub *) final;
  void visit(const luci::CircleSum *) final;
  void visit(const luci::CircleTanh *) final;
  void visit(const luci::CircleTile *) final;
  void visit(const luci::CircleTopKV2 *) final;
  void visit(const luci::CircleTranspose *) final;
  void visit(const luci::CircleTransposeConv *) final;
  void visit(const luci::CircleUnidirectionalSequenceLSTM *) final;
  void visit(const luci::CircleUnique *) final;
  void visit(const luci::CircleUnpack *) final;
  void visit(const luci::CircleWhere *) final;
  void visit(const luci::CircleWhile *) final;
  void visit(const luci::CircleZerosLike *) final;

  // Circle Only
  void visit(const luci::CircleBCQFullyConnected *) final;
  void visit(const luci::CircleBCQGather *) final;
  void visit(const luci::CircleGRU *) final;
  void visit(const luci::CircleInstanceNorm *) final;

  // NOTE CircleInput and CircleOutput are not handled here as these need
  //      link with graph I/O

  // Virtual
  void visit(const luci::CircleCustomOut *) final;
  void visit(const luci::CircleIfOut *) final;
  // void visit(const luci::CircleInput *) final;
  void visit(const luci::CircleNonMaxSuppressionV4Out *) final;
  void visit(const luci::CircleNonMaxSuppressionV5Out *) final;
  // void visit(const luci::CircleOutput *) final;
  void visit(const luci::CircleOutputDummy *) final;
  void visit(const luci::CircleOutputExclude *) final;
  void visit(const luci::CircleSplitOut *) final;
  void visit(const luci::CircleSplitVOut *) final;
  void visit(const luci::CircleTopKV2Out *) final;
  void visit(const luci::CircleUniqueOut *) final;
  void visit(const luci::CircleUnpackOut *) final;
  void visit(const luci::CircleVariable *) final;
  void visit(const luci::CircleWhileOut *) final;

public:
  luci::CircleNode *find_clone(const luci::CircleNode *node);

protected:
  luci::CloneContext &_clonecontext;
};

/**
 * @brief Connect cloned node from input node
 */
void clone_connect(const luci::CircleNode *node, luci::CloneContext &clonecontext);

} // namespace luci

#endif // __LUCI_PARTITION_CONNECT_NODE_H__
