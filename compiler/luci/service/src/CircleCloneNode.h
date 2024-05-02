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

#ifndef __CIRCLE_CLONE_NODE_H__
#define __CIRCLE_CLONE_NODE_H__

#include <luci/IR/CircleNodes.h>

#include <luci/IR/CircleNodeVisitor.h>

namespace luci
{

// CloneNode-let type
enum class CN
{
  ABC,
  DEF,
  GHIJ,
  KLMN,
  OPQR,
  STUV,
  WXYZ,
};

template <CN ct> class CloneNodeLet;

template <> class CloneNodeLet<CN::ABC> final : public luci::CircleNodeVisitor<luci::CircleNode *>
{
public:
  CloneNodeLet(loco::Graph *graph) : _graph(graph){};

public:
  luci::CircleNode *visit(const luci::CircleAbs *) final;
  luci::CircleNode *visit(const luci::CircleAdd *) final;
  luci::CircleNode *visit(const luci::CircleAddN *) final;
  luci::CircleNode *visit(const luci::CircleArgMax *) final;
  luci::CircleNode *visit(const luci::CircleArgMin *) final;
  luci::CircleNode *visit(const luci::CircleAveragePool2D *) final;
  luci::CircleNode *visit(const luci::CircleBatchMatMul *) final;
  luci::CircleNode *visit(const luci::CircleBatchToSpaceND *) final;
  luci::CircleNode *visit(const luci::CircleBroadcastTo *) final;
  luci::CircleNode *visit(const luci::CircleCast *) final;
  luci::CircleNode *visit(const luci::CircleCeil *) final;
  luci::CircleNode *visit(const luci::CircleConcatenation *) final;
  luci::CircleNode *visit(const luci::CircleConst *) final;
  luci::CircleNode *visit(const luci::CircleConv2D *) final;
  luci::CircleNode *visit(const luci::CircleCos *) final;
  luci::CircleNode *visit(const luci::CircleCumSum *) final;
  luci::CircleNode *visit(const luci::CircleCustom *) final;

  luci::CircleNode *visit(const luci::CircleNode *) final { return nullptr; }

protected:
  loco::Graph *_graph = nullptr;
};

template <> class CloneNodeLet<CN::DEF> final : public luci::CircleNodeVisitor<luci::CircleNode *>
{
public:
  CloneNodeLet(loco::Graph *graph) : _graph(graph){};

public:
  luci::CircleNode *visit(const luci::CircleDensify *) final;
  luci::CircleNode *visit(const luci::CircleDepthToSpace *) final;
  luci::CircleNode *visit(const luci::CircleDepthwiseConv2D *) final;
  luci::CircleNode *visit(const luci::CircleDequantize *) final;
  luci::CircleNode *visit(const luci::CircleDiv *) final;
  luci::CircleNode *visit(const luci::CircleElu *) final;
  luci::CircleNode *visit(const luci::CircleEqual *) final;
  luci::CircleNode *visit(const luci::CircleExp *) final;
  luci::CircleNode *visit(const luci::CircleExpandDims *) final;
  luci::CircleNode *visit(const luci::CircleFakeQuant *) final;
  luci::CircleNode *visit(const luci::CircleFill *) final;
  luci::CircleNode *visit(const luci::CircleFloor *) final;
  luci::CircleNode *visit(const luci::CircleFloorDiv *) final;
  luci::CircleNode *visit(const luci::CircleFloorMod *) final;
  luci::CircleNode *visit(const luci::CircleFullyConnected *) final;

  luci::CircleNode *visit(const luci::CircleNode *) final { return nullptr; }

protected:
  loco::Graph *_graph = nullptr;
};

template <> class CloneNodeLet<CN::GHIJ> final : public luci::CircleNodeVisitor<luci::CircleNode *>
{
public:
  CloneNodeLet(loco::Graph *graph) : _graph(graph){};

public:
  luci::CircleNode *visit(const luci::CircleGather *) final;
  luci::CircleNode *visit(const luci::CircleGatherNd *) final;
  luci::CircleNode *visit(const luci::CircleGelu *) final;
  luci::CircleNode *visit(const luci::CircleGreater *) final;
  luci::CircleNode *visit(const luci::CircleGreaterEqual *) final;
  luci::CircleNode *visit(const luci::CircleHardSwish *) final;
  luci::CircleNode *visit(const luci::CircleIf *) final;

  luci::CircleNode *visit(const luci::CircleNode *) final { return nullptr; }

protected:
  loco::Graph *_graph = nullptr;
};

template <> class CloneNodeLet<CN::KLMN> final : public luci::CircleNodeVisitor<luci::CircleNode *>
{
public:
  CloneNodeLet(loco::Graph *graph) : _graph(graph){};

public:
  luci::CircleNode *visit(const luci::CircleL2Normalize *) final;
  luci::CircleNode *visit(const luci::CircleL2Pool2D *) final;
  luci::CircleNode *visit(const luci::CircleLeakyRelu *) final;
  luci::CircleNode *visit(const luci::CircleLess *) final;
  luci::CircleNode *visit(const luci::CircleLessEqual *) final;
  luci::CircleNode *visit(const luci::CircleLocalResponseNormalization *) final;
  luci::CircleNode *visit(const luci::CircleLog *) final;
  luci::CircleNode *visit(const luci::CircleLogicalAnd *) final;
  luci::CircleNode *visit(const luci::CircleLogicalNot *) final;
  luci::CircleNode *visit(const luci::CircleLogicalOr *) final;
  luci::CircleNode *visit(const luci::CircleLogistic *) final;
  luci::CircleNode *visit(const luci::CircleLogSoftmax *) final;
  luci::CircleNode *visit(const luci::CircleMatrixDiag *) final;
  luci::CircleNode *visit(const luci::CircleMatrixSetDiag *) final;
  luci::CircleNode *visit(const luci::CircleMaximum *) final;
  luci::CircleNode *visit(const luci::CircleMaxPool2D *) final;
  luci::CircleNode *visit(const luci::CircleMean *) final;
  luci::CircleNode *visit(const luci::CircleMinimum *) final;
  luci::CircleNode *visit(const luci::CircleMirrorPad *) final;
  luci::CircleNode *visit(const luci::CircleMul *) final;
  luci::CircleNode *visit(const luci::CircleNeg *) final;
  luci::CircleNode *visit(const luci::CircleNonMaxSuppressionV4 *) final;
  luci::CircleNode *visit(const luci::CircleNonMaxSuppressionV5 *) final;
  luci::CircleNode *visit(const luci::CircleNotEqual *) final;

  luci::CircleNode *visit(const luci::CircleNode *) final { return nullptr; }

protected:
  loco::Graph *_graph = nullptr;
};

template <> class CloneNodeLet<CN::OPQR> final : public luci::CircleNodeVisitor<luci::CircleNode *>
{
public:
  CloneNodeLet(loco::Graph *graph) : _graph(graph){};

public:
  luci::CircleNode *visit(const luci::CircleOneHot *) final;
  luci::CircleNode *visit(const luci::CirclePack *) final;
  luci::CircleNode *visit(const luci::CirclePad *) final;
  luci::CircleNode *visit(const luci::CirclePadV2 *) final;
  luci::CircleNode *visit(const luci::CirclePow *) final;
  luci::CircleNode *visit(const luci::CirclePRelu *) final;
  luci::CircleNode *visit(const luci::CircleQuantize *) final;
  luci::CircleNode *visit(const luci::CircleRange *) final;
  luci::CircleNode *visit(const luci::CircleRank *) final;
  luci::CircleNode *visit(const luci::CircleReduceAny *) final;
  luci::CircleNode *visit(const luci::CircleReduceMax *) final;
  luci::CircleNode *visit(const luci::CircleReduceMin *) final;
  luci::CircleNode *visit(const luci::CircleReduceProd *) final;
  luci::CircleNode *visit(const luci::CircleRelu *) final;
  luci::CircleNode *visit(const luci::CircleRelu0To1 *) final;
  luci::CircleNode *visit(const luci::CircleRelu6 *) final;
  luci::CircleNode *visit(const luci::CircleReluN1To1 *) final;
  luci::CircleNode *visit(const luci::CircleReshape *) final;
  luci::CircleNode *visit(const luci::CircleResizeBilinear *) final;
  luci::CircleNode *visit(const luci::CircleResizeNearestNeighbor *) final;
  luci::CircleNode *visit(const luci::CircleReverseSequence *) final;
  luci::CircleNode *visit(const luci::CircleReverseV2 *) final;
  luci::CircleNode *visit(const luci::CircleRound *) final;
  luci::CircleNode *visit(const luci::CircleRsqrt *) final;

  luci::CircleNode *visit(const luci::CircleNode *) final { return nullptr; }

protected:
  loco::Graph *_graph = nullptr;
};

template <> class CloneNodeLet<CN::STUV> final : public luci::CircleNodeVisitor<luci::CircleNode *>
{
public:
  CloneNodeLet(loco::Graph *graph) : _graph(graph){};

public:
  luci::CircleNode *visit(const luci::CircleScatterNd *) final;
  luci::CircleNode *visit(const luci::CircleSegmentSum *) final;
  luci::CircleNode *visit(const luci::CircleSelect *) final;
  luci::CircleNode *visit(const luci::CircleSelectV2 *) final;
  luci::CircleNode *visit(const luci::CircleShape *) final;
  luci::CircleNode *visit(const luci::CircleSin *) final;
  luci::CircleNode *visit(const luci::CircleSlice *) final;
  luci::CircleNode *visit(const luci::CircleSoftmax *) final;
  luci::CircleNode *visit(const luci::CircleSpaceToBatchND *) final;
  luci::CircleNode *visit(const luci::CircleSpaceToDepth *) final;
  luci::CircleNode *visit(const luci::CircleSparseToDense *) final;
  luci::CircleNode *visit(const luci::CircleSplit *) final;
  luci::CircleNode *visit(const luci::CircleSplitV *) final;
  luci::CircleNode *visit(const luci::CircleSqrt *) final;
  luci::CircleNode *visit(const luci::CircleSquare *) final;
  luci::CircleNode *visit(const luci::CircleSquaredDifference *) final;
  luci::CircleNode *visit(const luci::CircleSqueeze *) final;
  luci::CircleNode *visit(const luci::CircleStridedSlice *) final;
  luci::CircleNode *visit(const luci::CircleSVDF *) final;
  luci::CircleNode *visit(const luci::CircleSub *) final;
  luci::CircleNode *visit(const luci::CircleSum *) final;
  luci::CircleNode *visit(const luci::CircleTanh *) final;
  luci::CircleNode *visit(const luci::CircleTile *) final;
  luci::CircleNode *visit(const luci::CircleTopKV2 *) final;
  luci::CircleNode *visit(const luci::CircleTranspose *) final;
  luci::CircleNode *visit(const luci::CircleTransposeConv *) final;
  luci::CircleNode *visit(const luci::CircleUnidirectionalSequenceLSTM *) final;
  luci::CircleNode *visit(const luci::CircleUnique *) final;
  luci::CircleNode *visit(const luci::CircleUnpack *) final;

  luci::CircleNode *visit(const luci::CircleNode *) final { return nullptr; }

protected:
  loco::Graph *_graph = nullptr;
};

template <> class CloneNodeLet<CN::WXYZ> final : public luci::CircleNodeVisitor<luci::CircleNode *>
{
public:
  CloneNodeLet(loco::Graph *graph) : _graph(graph){};

public:
  luci::CircleNode *visit(const luci::CircleWhere *) final;
  luci::CircleNode *visit(const luci::CircleWhile *) final;
  luci::CircleNode *visit(const luci::CircleZerosLike *) final;

  luci::CircleNode *visit(const luci::CircleNode *) final { return nullptr; }

protected:
  loco::Graph *_graph = nullptr;
};

class CloneNode final : public luci::CircleNodeVisitor<luci::CircleNode *>
{
public:
  CloneNode(loco::Graph *graph) : _graph(graph){};

public:
  // Circle Only
  luci::CircleNode *visit(const luci::CircleBCQFullyConnected *) final;
  luci::CircleNode *visit(const luci::CircleBCQGather *) final;
  luci::CircleNode *visit(const luci::CircleInstanceNorm *) final;
  luci::CircleNode *visit(const luci::CircleGRU *) final;

  // NOTE CircleInput and CircleOutput are not handled here as these need
  //      link with graph I/O

  // Virtual
  luci::CircleNode *visit(const luci::CircleCustomOut *) final;
  luci::CircleNode *visit(const luci::CircleIfOut *) final;
  // luci::CircleNode *visit(const luci::CircleInput *) final;
  luci::CircleNode *visit(const luci::CircleNonMaxSuppressionV4Out *) final;
  luci::CircleNode *visit(const luci::CircleNonMaxSuppressionV5Out *) final;
  // luci::CircleNode *visit(const luci::CircleOutput *) final;
  luci::CircleNode *visit(const luci::CircleOutputDummy *) final;
  luci::CircleNode *visit(const luci::CircleOutputExclude *) final;
  luci::CircleNode *visit(const luci::CircleSplitOut *) final;
  luci::CircleNode *visit(const luci::CircleSplitVOut *) final;
  luci::CircleNode *visit(const luci::CircleTopKV2Out *) final;
  luci::CircleNode *visit(const luci::CircleUniqueOut *) final;
  luci::CircleNode *visit(const luci::CircleUnpackOut *) final;
  luci::CircleNode *visit(const luci::CircleVariable *) final;
  luci::CircleNode *visit(const luci::CircleWhileOut *) final;

  // Handle in CircleNode
  luci::CircleNode *visit(const luci::CircleNode *) final;

  // NOTE CircleNodeVisitor will throw if not supported here

protected:
  loco::Graph *_graph = nullptr;
};

} // namespace luci

#endif // __CIRCLE_CLONE_NODE_H__
