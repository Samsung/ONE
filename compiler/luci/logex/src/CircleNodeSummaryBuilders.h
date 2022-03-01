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

class CircleBidirectionalSequenceLSTMOutSummaryBuilder final
  : public CircleNodeWithINPUTSummaryBuilder
{
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

class CircleCustomSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *node);
  void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
};

class CircleCustomOutSummaryBuilder final : public CircleNodeWithINPUTSummaryBuilder
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

class CircleGreaterSummaryBuilder final : public CircleNodeWithXYSummaryBuilder
{
};

class CircleGreaterEqualSummaryBuilder final : public CircleNodeWithXYSummaryBuilder
{
};

class CircleIfSummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *node);
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

class CircleNonMaxSuppressionV4OutSummaryBuilder final : public CircleNodeWithINPUTSummaryBuilder
{
};

class CircleNonMaxSuppressionV5SummaryBuilder final : public CircleNodeSummaryBuilder
{
private:
  std::vector<std::string> get_input_names(const luci::CircleNode *);
};

class CircleNonMaxSuppressionV5OutSummaryBuilder final : public CircleNodeWithINPUTSummaryBuilder
{
};

class CircleNotEqualSummaryBuilder final : public CircleNodeWithXYSummaryBuilder
{
};

} // namespace luci

#endif // __LUCI_LOGEX_CIRCLE_NODE_SUMMARY_BUILDERS__
