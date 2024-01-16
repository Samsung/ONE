/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License")
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

#include "CircleNodeSummaryBuilder.h"
#include "CircleNodeSummaryBuilders.h"

#include <luci/IR/CircleDialect.h>

#include <memory>

namespace
{

std::string circle_opname(luci::CircleOpcode opcode)
{
  static const std::string prefix{"circle."};

  switch (opcode)
  {
#define CIRCLE_NODE(OPCODE, CLASS) \
  case luci::CircleOpcode::OPCODE: \
    return prefix + #OPCODE;
#define CIRCLE_VNODE CIRCLE_NODE
#include <luci/IR/CircleNodes.lst>
#undef CIRCLE_VNODE
#undef CIRCLE_NODE
    default:
      break;
  };

  return prefix + "Invalid";
}

} // namespace

namespace luci
{

bool CircleNodeSummaryBuilder::build(const loco::Node *node, const locop::SymbolTable *tbl,
                                     locop::NodeSummary &s)
{
  if (node->dialect() != luci::CircleDialect::get())
    return false;

  auto ptr_to_str = [](const void *ptr) {
    std::stringstream ss;
    ss << ptr;
    return ss.str();
  };

  auto shape_to_str = [](const luci::CircleNode *node) {
    std::stringstream ss;
    ss << "<";
    for (uint32_t i = 0; i < node->rank(); ++i)
    {
      if (i)
        ss << ",";
      ss << (node->dim(i).known() ? node->dim(i).value() : -1);
    }
    ss << ">";
    return ss.str();
  };

  auto circle_node = loco::must_cast<const luci::CircleNode *>(node);
  if (const auto builder = create_builder(circle_node))
  {
    if (!builder->validate(circle_node))
    {
      s.state(locop::NodeDesc::State::Invalid);
      return false;
    }

    auto input_names = builder->get_input_names(circle_node);
    assert(node->arity() == input_names.size());
    for (uint32_t i = 0; i < node->arity(); ++i)
      s.args().append(input_names.at(i), tbl->lookup(node->arg(i)));

    builder->build_attributes(circle_node, s);
    builder->update_status(s);

    s.opname(circle_opname(circle_node->opcode()));
    s.comments().append("[" + circle_node->name() + " " + shape_to_str(circle_node) +
                        "] = " + ptr_to_str(node));

    return true;
  }
  else
  {
    // When SummaryBuilder is not implemented, return false
    return false;
  }
}

bool CircleNodeSummaryBuilder::validate(const luci::CircleNode *) { return true; }

std::vector<std::string> CircleNodeSummaryBuilder::get_input_names(const luci::CircleNode *)
{
  // Return empty names for default
  return std::vector<std::string>();
}

void CircleNodeSummaryBuilder::build_attributes(const luci::CircleNode *, locop::NodeSummary &)
{
  // Do nothing for default
}

void CircleNodeSummaryBuilder::update_status(locop::NodeSummary &s)
{
  s.state(locop::NodeDesc::State::Complete);
}

std::unique_ptr<CircleNodeSummaryBuilder>
CircleNodeSummaryBuilder::create_builder(const luci::CircleNode *node)
{
  switch (node->opcode())
  {
#define CIRCLE_NODE(OPCODE, CLASS)    \
  case luci::CircleOpcode::OPCODE:    \
  {                                   \
    return std::make_unique<CLASS>(); \
  }

    CIRCLE_NODE(ABS, CircleAbsSummaryBuilder)
    CIRCLE_NODE(ADD, CircleAddSummaryBuilder)
    CIRCLE_NODE(ADD_N, CircleAddNSummaryBuilder)
    CIRCLE_NODE(ARG_MAX, CircleArgMaxSummaryBuilder)
    CIRCLE_NODE(ARG_MIN, CircleArgMinSummaryBuilder)
    CIRCLE_NODE(AVERAGE_POOL_2D, CircleAveragePool2DSummaryBuilder)
    CIRCLE_NODE(BATCH_MATMUL, CircleBatchMatMulSummaryBuilder)
    CIRCLE_NODE(BATCH_TO_SPACE_ND, CircleBatchToSpaceNDSummaryBuilder)
    CIRCLE_NODE(BCQ_FULLY_CONNECTED, CircleBCQFullyConnectedSummaryBuilder)
    CIRCLE_NODE(BCQ_GATHER, CircleBCQGatherSummaryBuilder)
    CIRCLE_NODE(BIDIRECTIONAL_SEQUENCE_LSTM, CircleBidirectionalSequenceLSTMSummaryBuilder)
    CIRCLE_NODE(BROADCAST_TO, CircleBroadcastToSummaryBuilder)
    CIRCLE_NODE(CAST, CircleCastSummaryBuilder)
    CIRCLE_NODE(CEIL, CircleCeilSummaryBuilder)
    CIRCLE_NODE(CONCATENATION, CircleConcatenationSummaryBuilder)
    CIRCLE_NODE(CIRCLECONST, CircleConstSummaryBuilder)
    CIRCLE_NODE(CONV_2D, CircleConv2DSummaryBuilder)
    CIRCLE_NODE(COS, CircleCosSummaryBuilder)
    CIRCLE_NODE(CUMSUM, CircleCumsumSummaryBuilder)
    CIRCLE_NODE(CUSTOM, CircleCustomSummaryBuilder)
    CIRCLE_NODE(DENSIFY, CircleDensifySummaryBuilder)
    CIRCLE_NODE(DEPTH_TO_SPACE, CircleDepthToSpaceSummaryBuilder)
    CIRCLE_NODE(DEPTHWISE_CONV_2D, CircleDepthwiseConv2DSummaryBuilder)
    CIRCLE_NODE(DEQUANTIZE, CircleDequantizeSummaryBuilder)
    CIRCLE_NODE(DIV, CircleDivSummaryBuilder)
    CIRCLE_NODE(ELU, CircleEluSummaryBuilder)
    CIRCLE_NODE(EQUAL, CircleEqualSummaryBuilder)
    CIRCLE_NODE(EXP, CircleExpSummaryBuilder)
    CIRCLE_NODE(EXPAND_DIMS, CircleExpandDimsSummaryBuilder)
    CIRCLE_NODE(FAKE_QUANT, CircleFakeQuantSummaryBuilder)
    CIRCLE_NODE(FILL, CircleFillSummaryBuilder)
    CIRCLE_NODE(FLOOR, CircleFloorSummaryBuilder)
    CIRCLE_NODE(FLOOR_DIV, CircleFloorDivSummaryBuilder)
    CIRCLE_NODE(FLOOR_MOD, CircleFloorModSummaryBuilder)
    CIRCLE_NODE(FULLY_CONNECTED, CircleFullyConnectedSummaryBuilder)
    CIRCLE_NODE(GATHER, CircleGatherSummaryBuilder)
    CIRCLE_NODE(GATHER_ND, CircleGatherNdSummaryBuilder)
    CIRCLE_NODE(GELU, CircleGeluSummaryBuilder)
    CIRCLE_NODE(GREATER, CircleGreaterSummaryBuilder)
    CIRCLE_NODE(GREATER_EQUAL, CircleGreaterEqualSummaryBuilder)
    CIRCLE_NODE(CIR_GRU, CircleCirGruSummaryBuilder)
    CIRCLE_NODE(HARD_SWISH, CircleHardSwishSummaryBuilder)
    CIRCLE_NODE(IF, CircleIfSummaryBuilder)
    CIRCLE_NODE(INSTANCE_NORM, CircleInstanceNormSummaryBuilder)
    CIRCLE_NODE(L2_NORMALIZATION, CircleL2NormalizeSummaryBuilder)
    CIRCLE_NODE(L2_POOL_2D, CircleL2Pool2DSummaryBuilder)
    CIRCLE_NODE(LEAKY_RELU, CircleLeakyReluSummaryBuilder)
    CIRCLE_NODE(LESS, CircleLessSummaryBuilder)
    CIRCLE_NODE(LESS_EQUAL, CircleLessEqualSummaryBuilder)
    CIRCLE_NODE(LOCAL_RESPONSE_NORMALIZATION, CircleLocalResponseNormalizationSummaryBuilder)
    CIRCLE_NODE(LOG, CircleLogSummaryBuilder)
    CIRCLE_NODE(LOGICAL_AND, CircleLogicalAndSummaryBuilder)
    CIRCLE_NODE(LOGICAL_NOT, CircleLogicalNotSummaryBuilder)
    CIRCLE_NODE(LOGICAL_OR, CircleLogicalOrSummaryBuilder)
    CIRCLE_NODE(LOGISTIC, CircleLogisticSummaryBuilder)
    CIRCLE_NODE(LOG_SOFTMAX, CircleLogSoftmaxSummaryBuilder)
    CIRCLE_NODE(MATRIX_DIAG, CircleMatrixDiagSummaryBuilder)
    CIRCLE_NODE(MATRIX_SET_DIAG, CircleMatrixSetDiagSummaryBuilder)
    CIRCLE_NODE(MAXIMUM, CircleMaximumSummaryBuilder)
    CIRCLE_NODE(MAX_POOL_2D, CircleMaxPool2DSummaryBuilder)
    CIRCLE_NODE(MEAN, CircleMeanSummaryBuilder)
    CIRCLE_NODE(MINIMUM, CircleMinimumSummaryBuilder)
    CIRCLE_NODE(MIRROR_PAD, CircleMirrorPadSummaryBuilder)
    CIRCLE_NODE(MUL, CircleMulSummaryBuilder)
    CIRCLE_NODE(NEG, CircleNegSummaryBuilder)
    CIRCLE_NODE(NON_MAX_SUPPRESSION_V4, CircleNonMaxSuppressionV4SummaryBuilder)
    CIRCLE_NODE(NON_MAX_SUPPRESSION_V5, CircleNonMaxSuppressionV5SummaryBuilder)
    CIRCLE_NODE(NOT_EQUAL, CircleNotEqualSummaryBuilder)
    CIRCLE_NODE(ONE_HOT, CircleOneHotSummaryBuilder)
    CIRCLE_NODE(PACK, CirclePackSummaryBuilder)
    CIRCLE_NODE(PAD, CirclePadSummaryBuilder)
    CIRCLE_NODE(PADV2, CirclePadV2SummaryBuilder)
    CIRCLE_NODE(POW, CirclePowSummaryBuilder)
    CIRCLE_NODE(PRELU, CirclePReluSummaryBuilder)
    CIRCLE_NODE(QUANTIZE, CircleQuantizeSummaryBuilder)
    CIRCLE_NODE(RANGE, CircleRangeSummaryBuilder)
    CIRCLE_NODE(RANK, CircleRankSummaryBuilder)
    CIRCLE_NODE(REDUCE_ANY, CircleReduceAnySummaryBuilder)
    CIRCLE_NODE(REDUCE_MAX, CircleReduceMaxSummaryBuilder)
    CIRCLE_NODE(REDUCE_MIN, CircleReduceMinSummaryBuilder)
    CIRCLE_NODE(REDUCE_PROD, CircleReduceProdSummaryBuilder)
    CIRCLE_NODE(RELU, CircleReluSummaryBuilder)
    CIRCLE_NODE(RELU6, CircleRelu6SummaryBuilder)
    CIRCLE_NODE(RELU_N1_TO_1, CircleReluN1To1SummaryBuilder)
    CIRCLE_NODE(RESHAPE, CircleReshapeSummaryBuilder)
    CIRCLE_NODE(RESIZE_BILINEAR, CircleResizeBilinearSummaryBuilder)
    CIRCLE_NODE(RESIZE_NEAREST_NEIGHBOR, CircleResizeNearestNeighborSummaryBuilder)
    CIRCLE_NODE(REVERSE_SEQUENCE, CircleReverseSequenceSummaryBuilder)
    CIRCLE_NODE(REVERSE_V2, CircleReverseV2SummaryBuilder)
    CIRCLE_NODE(ROUND, CircleRoundSummaryBuilder)
    CIRCLE_NODE(RSQRT, CircleRsqrtSummaryBuilder)
    CIRCLE_NODE(SCATTER_ND, CircleScatterNdSummaryBuilder)
    CIRCLE_NODE(SEGMENT_SUM, CircleSegmentSumSummaryBuilder)
    CIRCLE_NODE(SELECT, CircleSelectSummaryBuilder)
    CIRCLE_NODE(SELECT_V2, CircleSelectV2SummaryBuilder)
    CIRCLE_NODE(SHAPE, CircleShapeSummaryBuilder)
    CIRCLE_NODE(SIN, CircleSinSummaryBuilder)
    CIRCLE_NODE(SLICE, CircleSliceSummaryBuilder)
    CIRCLE_NODE(SOFTMAX, CircleSoftmaxSummaryBuilder)
    CIRCLE_NODE(SPACE_TO_BATCH_ND, CircleSpaceToBatchNDSummaryBuilder)
    CIRCLE_NODE(SPACE_TO_DEPTH, CircleSpaceToDepthSummaryBuilder)
    CIRCLE_NODE(SPARSE_TO_DENSE, CircleSparseToDenseSummaryBuilder)
    CIRCLE_NODE(SPLIT, CircleSplitSummaryBuilder)
    CIRCLE_NODE(SPLIT_V, CircleSplitVSummaryBuilder)
    CIRCLE_NODE(SQRT, CircleSqrtSummaryBuilder)
    CIRCLE_NODE(SQUARE, CircleSquareSummaryBuilder)
    CIRCLE_NODE(SQUARED_DIFFERENCE, CircleSquaredDifferenceSummaryBuilder)
    CIRCLE_NODE(SQUEEZE, CircleSqueezeSummaryBuilder)
    CIRCLE_NODE(STRIDED_SLICE, CircleStridedSliceSummaryBuilder)
    CIRCLE_NODE(SUB, CircleSubSummaryBuilder)
    CIRCLE_NODE(SUM, CircleSumSummaryBuilder)
    CIRCLE_NODE(SVDF, CircleSVDFSummaryBuilder)
    CIRCLE_NODE(TANH, CircleTanhSummaryBuilder)
    CIRCLE_NODE(TILE, CircleTileSummaryBuilder)
    CIRCLE_NODE(TOPK_V2, CircleTopKV2SummaryBuilder)
    CIRCLE_NODE(TRANSPOSE, CircleTransposeSummaryBuilder)
    CIRCLE_NODE(TRANSPOSE_CONV, CircleTransposeConvSummaryBuilder)
    CIRCLE_NODE(UNIDIRECTIONAL_SEQUENCE_LSTM, CircleUnidirectionalSequenceLSTMSummaryBuilder)
    CIRCLE_NODE(UNIQUE, CircleUniqueSummaryBuilder)
    CIRCLE_NODE(UNPACK, CircleUnpackSummaryBuilder)
    CIRCLE_NODE(WHERE, CircleWhereSummaryBuilder)
    CIRCLE_NODE(WHILE, CircleWhileSummaryBuilder)
    CIRCLE_NODE(ZEROS_LIKE, CircleZerosLikeSummaryBuilder)

    CIRCLE_NODE(CIRCLEBIDIRECTIONAL_SEQUENCE_LSTM_OUT,
                CircleBidirectionalSequenceLSTMOutSummaryBuilder)
    CIRCLE_NODE(CIRCLECUSTOMOUT, CircleCustomOutSummaryBuilder)
    CIRCLE_NODE(CIRCLEIFOUT, CircleIfOutSummaryBuilder)
    CIRCLE_NODE(CIRCLEINPUT, CircleInputSummaryBuilder)
    CIRCLE_NODE(CIRCLENONMAXSUPPRESSIONV4OUT, CircleNonMaxSuppressionV4OutSummaryBuilder)
    CIRCLE_NODE(CIRCLENONMAXSUPPRESSIONV5OUT, CircleNonMaxSuppressionV5OutSummaryBuilder)
    CIRCLE_NODE(CIRCLEOUTPUT, CircleOutputSummaryBuilder)
    CIRCLE_NODE(CIRCLEOUTPUTDUMMY, CircleOutputDummySummaryBuilder)
    CIRCLE_NODE(CIRCLEOUTPUTEXCLUDE, CircleOutputExcludeSummaryBuilder)
    CIRCLE_NODE(CIRCLESPLITOUT, CircleSplitOutSummaryBuilder)
    CIRCLE_NODE(CIRCLESPLITVOUT, CircleSplitVOutSummaryBuilder)
    CIRCLE_NODE(CIRCLETOPKV2OUT, CircleTopKV2OutSummaryBuilder)
    CIRCLE_NODE(CIRCLEUNIQUEOUT, CircleUniqueOutSummaryBuilder)
    CIRCLE_NODE(CIRCLEUNPACKOUT, CircleUnpackOutSummaryBuilder)
    CIRCLE_NODE(CIRCLEVARIABLE, CircleVariableSummaryBuilder)
    CIRCLE_NODE(CIRCLEWHILEOUT, CircleWhileOutSummaryBuilder)

    default:
      return nullptr;

#undef CIRCLE_NODE
  }
}

} // namespace luci
