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

#ifndef __LUCI_LOGEX_CIRCLE_NODE_SUMMARY_BUILDERS__
#define __LUCI_LOGEX_CIRCLE_NODE_SUMMARY_BUILDERS__

#include "CircleNodeSummaryBuilder.h"

#include <luci/IR/CircleNode.h>

#include <string>
#include <vector>

namespace luci
{

class CircleNodeWithXSummaryBuilder : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
};

class CircleNodeWithINPUTSummaryBuilder : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
};

class CircleNodeWithXYSummaryBuilder : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
};

class CircleNodeWithFEATURESSummaryBuilder : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
};

template <class REDUCER_NODE>
class CircleNodeWithReducerSummaryBuilder : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *)
  {
    return {"input", "reduction_indices"};
  }

  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s)
  {
    auto mean = loco::must_cast<const REDUCER_NODE *>(node);
    s.args().append("keep_dims", mean->keep_dims() ? "true" : "false");
  }
};

} // namespace luci

namespace luci
{

class CircleAbsSummaryBuilder final : public CircleNodeWithXSummaryBuilder
{
};

class CircleAddSummaryBuilder final : public CircleNodeWithXYSummaryBuilder
{
private:
  bool validate(const luci::CircleNode *node);
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleAddNSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *node);
};

class CircleArgMaxSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleArgMinSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleAveragePool2DSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  bool validate(const luci::CircleNode *node);
  std::vector<std::string> get_input_names(const luci::CircleNode *);
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleBatchMatMulSummaryBuilder final : public CircleNodeWithXYSummaryBuilder
{
private:
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleBatchToSpaceNDSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
};

class CircleBCQFullyConnectedSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  bool validate(const luci::CircleNode *node);
  std::vector<std::string> get_input_names(const luci::CircleNode *);
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleBCQGatherSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleBidirectionalSequenceLSTMSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleBroadcastToSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
};

class CircleCastSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleCeilSummaryBuilder final : public CircleNodeWithXSummaryBuilder
{
};

class CircleConcatenationSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  bool validate(const luci::CircleNode *node);
  std::vector<std::string> get_input_names(const luci::CircleNode *node);
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleConstSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
  void update_status(locop::NodeSummary &s);
};

class CircleConv2DSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  bool validate(const luci::CircleNode *node);
  std::vector<std::string> get_input_names(const luci::CircleNode *);
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleCosSummaryBuilder final : public CircleNodeWithXSummaryBuilder
{
};

class CircleCumsumSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleCustomSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *node);
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleDensifySummaryBuilder final : public CircleNodeWithINPUTSummaryBuilder
{
};

class CircleDepthToSpaceSummaryBuilder final : public CircleNodeWithINPUTSummaryBuilder
{
private:
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleDepthwiseConv2DSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  bool validate(const luci::CircleNode *node);
  std::vector<std::string> get_input_names(const luci::CircleNode *);
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleDequantizeSummaryBuilder final : public CircleNodeWithINPUTSummaryBuilder
{
};

class CircleDivSummaryBuilder final : public CircleNodeWithXYSummaryBuilder
{
};

class CircleEluSummaryBuilder final : public CircleNodeWithFEATURESSummaryBuilder
{
};

class CircleEqualSummaryBuilder final : public CircleNodeWithXYSummaryBuilder
{
};

class CircleExpSummaryBuilder final : public CircleNodeWithXSummaryBuilder
{
};

class CircleExpandDimsSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
};

class CircleFakeQuantSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleFillSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
};

class CircleFloorSummaryBuilder final : public CircleNodeWithXSummaryBuilder
{
};

class CircleFloorDivSummaryBuilder final : public CircleNodeWithXYSummaryBuilder
{
};

class CircleFloorModSummaryBuilder final : public CircleNodeWithXYSummaryBuilder
{
};

class CircleFullyConnectedSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  bool validate(const luci::CircleNode *node);
  std::vector<std::string> get_input_names(const luci::CircleNode *);
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleGatherSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleGatherNdSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
};

class CircleGeluSummaryBuilder final : public CircleNodeWithFEATURESSummaryBuilder
{
private:
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleGreaterSummaryBuilder final : public CircleNodeWithXYSummaryBuilder
{
};

class CircleGreaterEqualSummaryBuilder final : public CircleNodeWithXYSummaryBuilder
{
};

class CircleGRUSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleHardSwishSummaryBuilder final : public CircleNodeWithFEATURESSummaryBuilder
{
};

class CircleIfSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *node);
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleInstanceNormSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  bool validate(const luci::CircleNode *node);
  std::vector<std::string> get_input_names(const luci::CircleNode *);
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleL2NormalizeSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  bool validate(const luci::CircleNode *node);
  std::vector<std::string> get_input_names(const luci::CircleNode *);
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleL2Pool2DSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  bool validate(const luci::CircleNode *node);
  std::vector<std::string> get_input_names(const luci::CircleNode *);
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleLeakyReluSummaryBuilder final : public CircleNodeWithFEATURESSummaryBuilder
{
private:
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleLessSummaryBuilder final : public CircleNodeWithXYSummaryBuilder
{
};

class CircleLessEqualSummaryBuilder final : public CircleNodeWithXYSummaryBuilder
{
};

class CircleLocalResponseNormalizationSummaryBuilder final
  : public CircleNodeWithINPUTSummaryBuilder
{
private:
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleLogSummaryBuilder final : public CircleNodeWithXSummaryBuilder
{
};

class CircleLogicalAndSummaryBuilder final : public CircleNodeWithXYSummaryBuilder
{
};

class CircleLogicalNotSummaryBuilder final : public CircleNodeWithXSummaryBuilder
{
};

class CircleLogicalOrSummaryBuilder final : public CircleNodeWithXYSummaryBuilder
{
};

class CircleLogisticSummaryBuilder final : public CircleNodeWithXSummaryBuilder
{
};

class CircleLogSoftmaxSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
};

class CircleMatrixDiagSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
};

class CircleMatrixSetDiagSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
};

class CircleMaximumSummaryBuilder final : public CircleNodeWithXYSummaryBuilder
{
};

class CircleMaxPool2DSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  bool validate(const luci::CircleNode *node);
  std::vector<std::string> get_input_names(const luci::CircleNode *);
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleMeanSummaryBuilder final : public CircleNodeWithReducerSummaryBuilder<luci::CircleMean>
{
};

class CircleMinimumSummaryBuilder final : public CircleNodeWithXYSummaryBuilder
{
};

class CircleMirrorPadSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  bool validate(const luci::CircleNode *node);
  std::vector<std::string> get_input_names(const luci::CircleNode *);
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleMulSummaryBuilder final : public CircleNodeWithXYSummaryBuilder
{
private:
  bool validate(const luci::CircleNode *node);
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleNegSummaryBuilder final : public CircleNodeWithXSummaryBuilder
{
};

class CircleNonMaxSuppressionV4SummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
};

class CircleNonMaxSuppressionV5SummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
};

class CircleNotEqualSummaryBuilder final : public CircleNodeWithXYSummaryBuilder
{
};

class CircleOneHotSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CirclePackSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *node);
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CirclePadSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
};

class CirclePadV2SummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
};

class CirclePowSummaryBuilder final : public CircleNodeWithXYSummaryBuilder
{
};

class CirclePReluSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
};

class CircleQuantizeSummaryBuilder final : public CircleNodeWithINPUTSummaryBuilder
{
};

class CircleRangeSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
};

class CircleRankSummaryBuilder final : public CircleNodeWithINPUTSummaryBuilder
{
};

class CircleReduceAnySummaryBuilder final
  : public CircleNodeWithReducerSummaryBuilder<luci::CircleReduceAny>
{
};

class CircleReduceMaxSummaryBuilder final
  : public CircleNodeWithReducerSummaryBuilder<luci::CircleReduceMax>
{
};

class CircleReduceMinSummaryBuilder final
  : public CircleNodeWithReducerSummaryBuilder<luci::CircleReduceMin>
{
};

class CircleReduceProdSummaryBuilder final
  : public CircleNodeWithReducerSummaryBuilder<luci::CircleReduceProd>
{
};

class CircleReluSummaryBuilder final : public CircleNodeWithFEATURESSummaryBuilder
{
};

class CircleRelu0To1SummaryBuilder final : public CircleNodeWithFEATURESSummaryBuilder
{
};

class CircleRelu6SummaryBuilder final : public CircleNodeWithFEATURESSummaryBuilder
{
};

class CircleReluN1To1SummaryBuilder final : public CircleNodeWithFEATURESSummaryBuilder
{
};

class CircleReshapeSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
  void update_status(locop::NodeSummary &s);
};

class CircleResizeBilinearSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleResizeNearestNeighborSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleReverseSequenceSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleReverseV2SummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
};

class CircleRoundSummaryBuilder final : public CircleNodeWithXSummaryBuilder
{
};

class CircleRsqrtSummaryBuilder final : public CircleNodeWithXSummaryBuilder
{
};

class CircleScatterNdSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
};

class CircleSegmentSumSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
};

class CircleSelectSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
};

class CircleSelectV2SummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
};

class CircleShapeSummaryBuilder final : public CircleNodeWithINPUTSummaryBuilder
{
private:
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleSinSummaryBuilder final : public CircleNodeWithXSummaryBuilder
{
};

class CircleSliceSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
};

class CircleSoftmaxSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleSpaceToBatchNDSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
};

class CircleSpaceToDepthSummaryBuilder final : public CircleNodeWithINPUTSummaryBuilder
{
private:
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleSparseToDenseSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleSplitSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleSplitVSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleSqrtSummaryBuilder final : public CircleNodeWithXSummaryBuilder
{
};

class CircleSquareSummaryBuilder final : public CircleNodeWithXSummaryBuilder
{
};

class CircleSquaredDifferenceSummaryBuilder final : public CircleNodeWithXYSummaryBuilder
{
};

class CircleSqueezeSummaryBuilder final : public CircleNodeWithINPUTSummaryBuilder
{
private:
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleStridedSliceSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleSubSummaryBuilder final : public CircleNodeWithXYSummaryBuilder
{
};

class CircleSumSummaryBuilder final : public CircleNodeWithReducerSummaryBuilder<luci::CircleSum>
{
};

class CircleSVDFSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  bool validate(const luci::CircleNode *node);
  std::vector<std::string> get_input_names(const luci::CircleNode *);

  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleTanhSummaryBuilder final : public CircleNodeWithXSummaryBuilder
{
};

class CircleTileSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
};

class CircleTopKV2SummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
};

class CircleTransposeSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
};

class CircleTransposeConvSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  bool validate(const luci::CircleNode *node);
  std::vector<std::string> get_input_names(const luci::CircleNode *);
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleUnidirectionalSequenceLSTMSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleUniqueSummaryBuilder final : public CircleNodeWithINPUTSummaryBuilder
{
private:
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleUnpackSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleWhereSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
};

class CircleWhileSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *node);
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleZerosLikeSummaryBuilder final : public CircleNodeWithINPUTSummaryBuilder
{
};

class CircleBidirectionalSequenceLSTMOutSummaryBuilder final
  : public CircleNodeWithINPUTSummaryBuilder
{
};

class CircleCustomOutSummaryBuilder final : public CircleNodeWithINPUTSummaryBuilder
{
};

class CircleIfOutSummaryBuilder final : public CircleNodeWithINPUTSummaryBuilder
{
};

class CircleInputSummaryBuilder final : public CircleNodeSummaryBuilder
{
};

class CircleNonMaxSuppressionV4OutSummaryBuilder final : public CircleNodeWithINPUTSummaryBuilder
{
};

class CircleNonMaxSuppressionV5OutSummaryBuilder final : public CircleNodeWithINPUTSummaryBuilder
{
};

class CircleOutputSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
};

class CircleOutputDummySummaryBuilder final : public CircleNodeSummaryBuilder
{
};

class CircleOutputExcludeSummaryBuilder final : public CircleNodeSummaryBuilder
{
};

class CircleSplitOutSummaryBuilder final : public CircleNodeWithINPUTSummaryBuilder
{
};

class CircleSplitVOutSummaryBuilder final : public CircleNodeWithINPUTSummaryBuilder
{
};

class CircleTopKV2OutSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
};

class CircleUniqueOutSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
};

class CircleUnpackOutSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
};

class CircleVariableSummaryBuilder final : public CircleNodeSummaryBuilder
{
};

class CircleWhileOutSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
};

} // namespace luci

#endif // __LUCI_LOGEX_CIRCLE_NODE_SUMMARY_BUILDERS__
