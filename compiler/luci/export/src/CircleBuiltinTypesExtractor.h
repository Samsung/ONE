/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __CIRCLE_BUILTIN_TYPES_EXTRACTOR_H__
#define __CIRCLE_BUILTIN_TYPES_EXTRACTOR_H__

#include "CircleExporterUtils.h"

#include <luci/IR/CircleNode.h>
#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeVisitor.h>

#include <flatbuffers/flexbuffers.h>

namespace luci
{

// NOTE Virtual nodes are not circle builtin operators.
//      Therefore, they are not defined here.
class BuiltinOptionsExtractor final
  : public luci::CircleNodeMutableVisitor<flatbuffers::Offset<void>>
{
public:
  BuiltinOptionsExtractor(flatbuffers::FlatBufferBuilder &builder) : _builder{builder}
  {
    // DO NOTHING
  }

public:
  flatbuffers::Offset<void> visit(luci::CircleAbs *)
  {
    return circle::CreateAbsOptions(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleAdd *node)
  {
    return circle::CreateAddOptions(_builder, to_circle_actfunc(node->fusedActivationFunction()))
      .Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleAddN *)
  {
    return circle::CreateAddNOptions(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleArgMax *node)
  {
    return circle::CreateArgMaxOptions(_builder, luci::to_circle_tensortype(node->output_type()))
      .Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleArgMin *node)
  {
    return circle::CreateArgMinOptions(_builder, luci::to_circle_tensortype(node->output_type()))
      .Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleAveragePool2D *node)
  {
    return circle::CreatePool2DOptions(_builder, getOpPadding(node->padding()), node->stride()->w(),
                                       node->stride()->h(), node->filter()->w(),
                                       node->filter()->h(),
                                       to_circle_actfunc(node->fusedActivationFunction()))
      .Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleBatchMatMul *node)
  {
    return circle::CreateBatchMatMulOptions(_builder, node->adj_x(), node->adj_y()).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleBatchToSpaceND *)
  {
    return circle::CreateBatchToSpaceNDOptions(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleBidirectionalSequenceLSTM *node)
  {
    return circle::CreateBidirectionalSequenceLSTMOptions(
             _builder, to_circle_actfunc(node->fusedActivationFunction()), node->cell_clip(),
             node->proj_clip(), node->merge_outputs(), node->time_major(),
             node->asymmetric_quantize_inputs())
      .Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleCast *node)
  {
    if (node->out_data_type() == loco::DataType::Unknown)
      return _no_option;
    else
      return circle::CreateCastOptions(_builder, luci::to_circle_tensortype(node->in_data_type()),
                                       luci::to_circle_tensortype(node->out_data_type()))
        .Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleCeil *) { return _no_option; }
  flatbuffers::Offset<void> visit(luci::CircleConcatenation *node)
  {
    return circle::CreateConcatenationOptions(_builder, node->axis(),
                                              to_circle_actfunc(node->fusedActivationFunction()))
      .Union();
  }
  // CircleConst is not virtual but not builtinOperator
  // flatbuffers::Offset<void> visit(luci::CircleConst *)
  flatbuffers::Offset<void> visit(luci::CircleConv2D *node)
  {
    return circle::CreateConv2DOptions(_builder, getOpPadding(node->padding()), node->stride()->w(),
                                       node->stride()->h(),
                                       to_circle_actfunc(node->fusedActivationFunction()),
                                       node->dilation()->w(), node->dilation()->h())
      .Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleCos *)
  {
    return circle::CreateCosOptions(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleCustom *) { return _no_option; }
  flatbuffers::Offset<void> visit(luci::CircleCumSum *node)
  {
    return circle::CreateCumsumOptions(_builder, node->exclusive(), node->reverse()).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleDensify *)
  {
    return circle::CreateDensifyOptions(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleDepthToSpace *node)
  {
    return circle::CreateDepthToSpaceOptions(_builder, node->block_size()).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleDepthwiseConv2D *node)
  {
    return circle::CreateDepthwiseConv2DOptions(
             _builder, getOpPadding(node->padding()), node->stride()->w(), node->stride()->h(),
             node->depthMultiplier(), to_circle_actfunc(node->fusedActivationFunction()),
             node->dilation()->w(), node->dilation()->h())
      .Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleDequantize *) { return _no_option; }
  flatbuffers::Offset<void> visit(luci::CircleDiv *node)
  {
    return circle::CreateDivOptions(_builder, to_circle_actfunc(node->fusedActivationFunction()))
      .Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleElu *) { return _no_option; }
  flatbuffers::Offset<void> visit(luci::CircleEqual *)
  {
    return circle::CreateEqualOptions(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleExp *)
  {
    return circle::CreateExpOptions(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleExpandDims *)
  {
    return circle::CreateExpandDimsOptions(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleFakeQuant *node)
  {
    return circle::CreateFakeQuantOptions(_builder, node->min(), node->max(), node->num_bits(),
                                          node->narrow_range())
      .Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleFill *)
  {
    return circle::CreateFillOptions(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleFloor *) { return _no_option; }
  flatbuffers::Offset<void> visit(luci::CircleFloorDiv *)
  {
    return circle::CreateFloorDivOptions(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleFloorMod *)
  {
    return circle::CreateFloorModOptions(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleFullyConnected *node)
  {
    return circle::CreateFullyConnectedOptions(
             _builder, to_circle_actfunc(node->fusedActivationFunction()),
             to_circle_weightsformat(node->weights_format()), node->keep_num_dims())
      .Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleGather *node)
  {
    return circle::CreateGatherOptions(_builder, node->axis()).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleGatherNd *)
  {
    return circle::CreateGatherNdOptions(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleGelu *node)
  {
    return circle::CreateGeluOptions(_builder, node->approximate()).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleGreater *)
  {
    return circle::CreateGreaterOptions(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleGreaterEqual *)
  {
    return circle::CreateGreaterEqualOptions(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleHardSwish *) { return _no_option; }
  flatbuffers::Offset<void> visit(luci::CircleIf *node)
  {
    return circle::CreateIfOptions(_builder, node->then_branch(), node->else_branch()).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleL2Normalize *node)
  {
    return circle::CreateL2NormOptions(_builder, to_circle_actfunc(node->fusedActivationFunction()))
      .Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleL2Pool2D *node)
  {
    return circle::CreatePool2DOptions(_builder, getOpPadding(node->padding()), node->stride()->w(),
                                       node->stride()->h(), node->filter()->w(),
                                       node->filter()->h(),
                                       to_circle_actfunc(node->fusedActivationFunction()))
      .Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleLeakyRelu *node)
  {
    return circle::CreateLeakyReluOptions(_builder, node->alpha()).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleLess *)
  {
    return circle::CreateLessOptions(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleLessEqual *)
  {
    return circle::CreateLessEqualOptions(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleLocalResponseNormalization *node)
  {
    return circle::CreateLocalResponseNormalizationOptions(_builder, node->radius(), node->bias(),
                                                           node->alpha(), node->beta())
      .Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleLog *) { return _no_option; }
  flatbuffers::Offset<void> visit(luci::CircleLogicalAnd *)
  {
    return circle::CreateLogicalAndOptions(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleLogicalNot *)
  {
    return circle::CreateLogicalNotOptions(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleLogicalOr *)
  {
    return circle::CreateLogicalOrOptions(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleLogistic *) { return _no_option; }
  flatbuffers::Offset<void> visit(luci::CircleLogSoftmax *)
  {
    return circle::CreateLogSoftmaxOptions(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleMatrixDiag *)
  {
    return circle::CreateMatrixDiagOptions(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleMatrixSetDiag *)
  {
    return circle::CreateMatrixSetDiagOptions(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleMaximum *)
  {
    return circle::CreateMaximumMinimumOptions(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleMaxPool2D *node)
  {
    return circle::CreatePool2DOptions(_builder, getOpPadding(node->padding()), node->stride()->w(),
                                       node->stride()->h(), node->filter()->w(),
                                       node->filter()->h(),
                                       to_circle_actfunc(node->fusedActivationFunction()))
      .Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleMean *node)
  {
    return circle::CreateReducerOptions(_builder, node->keep_dims()).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleMinimum *)
  {
    return circle::CreateMaximumMinimumOptions(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleMirrorPad *node)
  {
    return circle::CreateMirrorPadOptions(_builder, to_circle_mirrorpadmode(node->mode())).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleMul *node)
  {
    return circle::CreateMulOptions(_builder, to_circle_actfunc(node->fusedActivationFunction()))
      .Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleNeg *)
  {
    return circle::CreateNegOptions(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleNonMaxSuppressionV4 *)
  {
    return circle::CreateNonMaxSuppressionV4Options(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleNonMaxSuppressionV5 *)
  {
    return circle::CreateNonMaxSuppressionV5Options(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleNotEqual *)
  {
    return circle::CreateNotEqualOptions(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleOneHot *node)
  {
    return circle::CreateOneHotOptions(_builder, node->axis()).Union();
  }
  flatbuffers::Offset<void> visit(luci::CirclePack *node)
  {
    return circle::CreatePackOptions(_builder, node->values_count(), node->axis()).Union();
  }
  flatbuffers::Offset<void> visit(luci::CirclePad *)
  {
    return circle::CreatePadOptions(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CirclePadV2 *)
  {
    return circle::CreatePadV2Options(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CirclePow *)
  {
    return circle::CreatePowOptions(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CirclePRelu *) { return _no_option; }
  flatbuffers::Offset<void> visit(luci::CircleQuantize *) { return _no_option; }
  flatbuffers::Offset<void> visit(luci::CircleRange *)
  {
    return circle::CreateRangeOptions(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleRank *)
  {
    return circle::CreateRankOptions(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleReduceAny *node)
  {
    return circle::CreateReducerOptions(_builder, node->keep_dims()).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleReduceMax *node)
  {
    return circle::CreateReducerOptions(_builder, node->keep_dims()).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleReduceMin *node)
  {
    return circle::CreateReducerOptions(_builder, node->keep_dims()).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleReduceProd *node)
  {
    return circle::CreateReducerOptions(_builder, node->keep_dims()).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleRelu *) { return _no_option; }
  flatbuffers::Offset<void> visit(luci::CircleRelu6 *) { return _no_option; }
  flatbuffers::Offset<void> visit(luci::CircleReluN1To1 *) { return _no_option; }
  flatbuffers::Offset<void> visit(luci::CircleReshape *node)
  {
    auto new_shape = _builder.CreateVector<int32_t>(
      node->newShape()->rank(), [node](size_t i) { return node->newShape()->dim(i); });
    return circle::CreateReshapeOptions(_builder, new_shape).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleResizeBilinear *node)
  {
    return circle::CreateResizeBilinearOptions(_builder, node->align_corners(),
                                               node->half_pixel_centers())
      .Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleResizeNearestNeighbor *node)
  {
    return circle::CreateResizeNearestNeighborOptions(_builder, node->align_corners()).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleReverseSequence *node)
  {
    return circle::CreateReverseSequenceOptions(_builder, node->seq_axis(), node->batch_axis())
      .Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleReverseV2 *)
  {
    return circle::CreateReverseV2Options(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleRound *) { return _no_option; }
  flatbuffers::Offset<void> visit(luci::CircleRsqrt *) { return _no_option; }
  flatbuffers::Offset<void> visit(luci::CircleScatterNd *)
  {
    return circle::CreateScatterNdOptions(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleSegmentSum *)
  {
    return circle::CreateSegmentSumOptions(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleSelect *)
  {
    return circle::CreateSelectOptions(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleSelectV2 *)
  {
    return circle::CreateSelectV2Options(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleShape *node)
  {
    return circle::CreateShapeOptions(_builder, luci::to_circle_tensortype(node->out_type()))
      .Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleSin *) { return _no_option; }
  flatbuffers::Offset<void> visit(luci::CircleSlice *)
  {
    return circle::CreateSliceOptions(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleSoftmax *node)
  {
    return circle::CreateSoftmaxOptions(_builder, node->beta()).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleSpaceToBatchND *)
  {
    return circle::CreateSpaceToBatchNDOptions(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleSpaceToDepth *node)
  {
    return circle::CreateSpaceToDepthOptions(_builder, node->block_size()).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleSparseToDense *node)
  {
    return circle::CreateSparseToDenseOptions(_builder, node->validate_indices()).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleSplit *node)
  {
    return circle::CreateSplitOptions(_builder, node->num_split()).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleSplitV *node)
  {
    return circle::CreateSplitVOptions(_builder, node->num_split()).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleSqrt *) { return _no_option; }
  flatbuffers::Offset<void> visit(luci::CircleSquare *)
  {
    return circle::CreateSquareOptions(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleSquaredDifference *)
  {
    return circle::CreateSquaredDifferenceOptions(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleSqueeze *node)
  {
    auto squeeze_dims = _builder.CreateVector<int32_t>(node->squeeze_dims());
    return circle::CreateSqueezeOptions(_builder, squeeze_dims).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleStridedSlice *node)
  {
    return circle::CreateStridedSliceOptions(_builder, node->begin_mask(), node->end_mask(),
                                             node->ellipsis_mask(), node->new_axis_mask(),
                                             node->shrink_axis_mask())
      .Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleSub *node)
  {
    return circle::CreateSubOptions(_builder, to_circle_actfunc(node->fusedActivationFunction()))
      .Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleSum *node)
  {
    return circle::CreateReducerOptions(_builder, node->keep_dims()).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleSVDF *node)
  {
    return circle::CreateSVDFOptions(_builder, node->svdf_rank(),
                                     to_circle_actfunc(node->fusedActivationFunction()),
                                     node->asymmetric_quantize_inputs())
      .Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleTanh *) { return _no_option; }
  flatbuffers::Offset<void> visit(luci::CircleTile *)
  {
    return circle::CreateTileOptions(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleTopKV2 *)
  {
    return circle::CreateTopKV2Options(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleTranspose *)
  {
    return circle::CreateTransposeOptions(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleTransposeConv *node)
  {
    return circle::CreateTransposeConvOptions(_builder, getOpPadding(node->padding()),
                                              node->stride()->w(), node->stride()->h(),
                                              to_circle_actfunc(node->fusedActivationFunction()))
      .Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleUnidirectionalSequenceLSTM *node)
  {
    return circle::CreateUnidirectionalSequenceLSTMOptions(
             _builder, to_circle_actfunc(node->fusedActivationFunction()), node->cell_clip(),
             node->proj_clip(), node->time_major(), node->asymmetric_quantize_inputs())
      .Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleUnique *node)
  {
    return circle::CreateUniqueOptions(_builder, luci::to_circle_tensortype(node->idx_out_type()))
      .Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleUnpack *node)
  {
    return circle::CreateUnpackOptions(_builder, node->num(), node->axis()).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleWhere *)
  {
    return circle::CreateWhereOptions(_builder).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleWhile *node)
  {
    return circle::CreateWhileOptions(_builder, node->cond_branch(), node->body_branch()).Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleZerosLike *)
  {
    return circle::CreateZerosLikeOptions(_builder).Union();
  }
  // Circle only
  flatbuffers::Offset<void> visit(luci::CircleBCQFullyConnected *node)
  {
    return circle::CreateBCQFullyConnectedOptions(
             _builder, node->weights_hidden_size(),
             to_circle_actfunc(node->fusedActivationFunction()))
      .Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleBCQGather *node)
  {
    return circle::CreateBCQGatherOptions(_builder, node->input_hidden_size(), node->axis())
      .Union();
  }
  flatbuffers::Offset<void> visit(luci::CircleInstanceNorm *node)
  {
    return circle::CreateInstanceNormOptions(_builder, node->epsilon(),
                                             to_circle_actfunc(node->fusedActivationFunction()))
      .Union();
  }

protected:
  flatbuffers::FlatBufferBuilder &_builder;

private:
  const flatbuffers::Offset<void> _no_option = 0;
};

} // namespace luci

#endif // __CIRCLE_BUILTIN_TYPES_EXTRACTOR_H__
