/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __TFLITE_OP_REGISTRY_H__
#define __TFLITE_OP_REGISTRY_H__

#include "TFliteOpChef.h"
#include "TFliteOpChefs.h"

#include <memory>

namespace tflchef
{

/**
 * @brief tflchef operator registry
 */
class TFliteOpRegistry
{
public:
  /**
   * @brief Returns registered TFliteOpChef pointer for BuiltinOperator or
   *        nullptr if not registered
   */
  const TFliteOpChef *lookup(tflite::BuiltinOperator op) const
  {
    if (_tfliteop_map.find(op) == _tfliteop_map.end())
      return nullptr;

    return _tfliteop_map.at(op).get();
  }

  static TFliteOpRegistry &get()
  {
    static TFliteOpRegistry me;
    return me;
  }

private:
  TFliteOpRegistry()
  {
#define REG_TFL_OP(OPCODE, CLASS) \
  _tfliteop_map[tflite::BuiltinOperator_##OPCODE] = std::make_unique<CLASS>()

    REG_TFL_OP(ABS, TFliteOpAbs);
    REG_TFL_OP(ADD, TFliteOpAdd);
    REG_TFL_OP(ADD_N, TFliteOpAddN);
    REG_TFL_OP(ARG_MAX, TFliteOpArgMax);
    REG_TFL_OP(ARG_MIN, TFliteOpArgMin);
    REG_TFL_OP(AVERAGE_POOL_2D, TFliteOpAveragePool2D);
    REG_TFL_OP(BATCH_MATMUL, TFliteOpBatchMatMul);
    REG_TFL_OP(BATCH_TO_SPACE_ND, TFliteOpBatchToSpaceND);
    REG_TFL_OP(BIDIRECTIONAL_SEQUENCE_LSTM, TFliteOpBidirectionalSequenceLSTM);
    REG_TFL_OP(CAST, TFliteOpCast);
    REG_TFL_OP(CEIL, TFliteOpCeil);
    REG_TFL_OP(CONCATENATION, TFliteOpConcatenation);
    REG_TFL_OP(CONV_2D, TFliteOpConv2D);
    REG_TFL_OP(COS, TFliteOpCos);
    REG_TFL_OP(CUMSUM, TFliteOpCumsum);
    REG_TFL_OP(DEPTH_TO_SPACE, TFliteOpDepthToSpace);
    REG_TFL_OP(DEPTHWISE_CONV_2D, TFliteOpDepthwiseConv2D);
    REG_TFL_OP(DEQUANTIZE, TFliteOpDequantize);
    REG_TFL_OP(DIV, TFliteOpDiv);
    REG_TFL_OP(ELU, TFliteOpELU);
    REG_TFL_OP(EQUAL, TFliteOpEqual);
    REG_TFL_OP(EXP, TFliteOpExp);
    REG_TFL_OP(EXPAND_DIMS, TFliteOpExpandDims);
    REG_TFL_OP(FAKE_QUANT, TFliteOpFakeQuant);
    REG_TFL_OP(FILL, TFliteOpFill);
    REG_TFL_OP(FLOOR, TFliteOpFloor);
    REG_TFL_OP(FLOOR_DIV, TFliteOpFloorDiv);
    REG_TFL_OP(FLOOR_MOD, TFliteOpFloorMod);
    REG_TFL_OP(FULLY_CONNECTED, TFliteOpFullyConnected);
    REG_TFL_OP(GATHER, TFliteOpGather);
    REG_TFL_OP(GATHER_ND, TFliteOpGatherNd);
    REG_TFL_OP(GELU, TFliteOpGelu);
    REG_TFL_OP(GREATER, TFliteOpGreater);
    REG_TFL_OP(GREATER_EQUAL, TFliteOpGreaterEqual);
    REG_TFL_OP(HARD_SWISH, TFliteOpHardSwish);
    REG_TFL_OP(L2_NORMALIZATION, TFliteOpL2Normalize);
    REG_TFL_OP(L2_POOL_2D, TFliteOpL2Pool2D);
    REG_TFL_OP(LEAKY_RELU, TFliteOpLeakyRelu);
    REG_TFL_OP(LESS, TFliteOpLess);
    REG_TFL_OP(LESS_EQUAL, TFliteOpLessEqual);
    REG_TFL_OP(LOCAL_RESPONSE_NORMALIZATION, TFliteOpLocalResponseNormalization);
    REG_TFL_OP(LOG, TFliteOpLog);
    REG_TFL_OP(LOGICAL_AND, TFliteOpLogicalAnd);
    REG_TFL_OP(LOGICAL_NOT, TFliteOpLogicalNot);
    REG_TFL_OP(LOGICAL_OR, TFliteOpLogicalOr);
    REG_TFL_OP(LOGISTIC, TFliteOpLogistic);
    REG_TFL_OP(LOG_SOFTMAX, TFliteOpLogSoftmax);
    REG_TFL_OP(MATRIX_DIAG, TFliteOpMatrixDiag);
    REG_TFL_OP(MAX_POOL_2D, TFliteOpMaxPool2D);
    REG_TFL_OP(MATRIX_SET_DIAG, TFliteOpMatrixSetDiag);
    REG_TFL_OP(MAXIMUM, TFliteOpMaximum);
    REG_TFL_OP(MEAN, TFliteOpMean);
    REG_TFL_OP(MINIMUM, TFliteOpMinimum);
    REG_TFL_OP(MIRROR_PAD, TFliteOpMirrorPad);
    REG_TFL_OP(MUL, TFliteOpMul);
    REG_TFL_OP(NEG, TFliteOpNeg);
    REG_TFL_OP(NON_MAX_SUPPRESSION_V4, TFliteOpNonMaxSuppressionV4);
    REG_TFL_OP(NON_MAX_SUPPRESSION_V5, TFliteOpNonMaxSuppressionV5);
    REG_TFL_OP(NOT_EQUAL, TFliteOpNotEqual);
    REG_TFL_OP(ONE_HOT, TFliteOpOneHot);
    REG_TFL_OP(PACK, TFliteOpPack);
    REG_TFL_OP(PAD, TFliteOpPad);
    REG_TFL_OP(PADV2, TFliteOpPadV2);
    REG_TFL_OP(POW, TFliteOpPow);
    REG_TFL_OP(PRELU, TFliteOpPRelu);
    REG_TFL_OP(QUANTIZE, TFliteOpQuantize);
    REG_TFL_OP(RANGE, TFliteOpRange);
    REG_TFL_OP(RANK, TFliteOpRank);
    REG_TFL_OP(REDUCE_ANY, TFliteOpReduceAny);
    REG_TFL_OP(REDUCE_MAX, TFliteOpReduceMax);
    REG_TFL_OP(REDUCE_MIN, TFliteOpReduceMin);
    REG_TFL_OP(REDUCE_PROD, TFliteOpReduceProd);
    REG_TFL_OP(RELU, TFliteOpReLU);
    REG_TFL_OP(RELU6, TFliteOpReLU6);
    REG_TFL_OP(RELU_N1_TO_1, TFliteOpReLUN1To1);
    REG_TFL_OP(RESHAPE, TFliteOpReshape);
    REG_TFL_OP(RESIZE_BILINEAR, TFliteOpResizeBilinear);
    REG_TFL_OP(RESIZE_NEAREST_NEIGHBOR, TFliteOpResizeNearestNeighbor);
    REG_TFL_OP(REVERSE_SEQUENCE, TFliteOpReverseSequence);
    REG_TFL_OP(REVERSE_V2, TFliteOpReverseV2);
    REG_TFL_OP(ROUND, TFliteOpRound);
    REG_TFL_OP(RSQRT, TFliteOpRsqrt);
    REG_TFL_OP(SCATTER_ND, TFliteOpScatterNd);
    REG_TFL_OP(SEGMENT_SUM, TFliteOpSegmentSum);
    REG_TFL_OP(SELECT, TFliteOpSelect);
    REG_TFL_OP(SELECT_V2, TFliteOpSelectV2);
    REG_TFL_OP(SHAPE, TFliteOpShape);
    REG_TFL_OP(SIN, TFliteOpSin);
    REG_TFL_OP(SLICE, TFliteOpSlice);
    REG_TFL_OP(SOFTMAX, TFliteOpSoftmax);
    REG_TFL_OP(SPACE_TO_BATCH_ND, TFliteOpSpaceToBatchND);
    REG_TFL_OP(SPACE_TO_DEPTH, TFliteOpSpaceToDepth);
    REG_TFL_OP(SPARSE_TO_DENSE, TFliteOpSparseToDense);
    REG_TFL_OP(SPLIT, TFliteOpSplit);
    REG_TFL_OP(SPLIT_V, TFliteOpSplitV);
    REG_TFL_OP(SQRT, TFliteOpSqrt);
    REG_TFL_OP(SQUARE, TFliteOpSquare);
    REG_TFL_OP(SQUARED_DIFFERENCE, TFliteOpSquaredDifference);
    REG_TFL_OP(SQUEEZE, TFliteOpSqueeze);
    REG_TFL_OP(STRIDED_SLICE, TFliteOpStridedSlice);
    REG_TFL_OP(SUB, TFliteOpSub);
    REG_TFL_OP(SUM, TFliteOpSum);
    REG_TFL_OP(SVDF, TFliteOpSVDF);
    REG_TFL_OP(TANH, TFliteOpTanh);
    REG_TFL_OP(TILE, TFliteOpTile);
    REG_TFL_OP(TOPK_V2, TFliteOpTopKV2);
    REG_TFL_OP(TRANSPOSE, TFliteOpTranspose);
    REG_TFL_OP(TRANSPOSE_CONV, TFliteOpTransposeConv);
    REG_TFL_OP(UNIDIRECTIONAL_SEQUENCE_LSTM, TFliteOpUnidirectionalSequenceLSTM);
    REG_TFL_OP(UNIQUE, TFliteOpUnique);
    REG_TFL_OP(UNPACK, TFliteOpUnpack);
    REG_TFL_OP(WHERE, TFliteOpWhere);
    REG_TFL_OP(ZEROS_LIKE, TFliteOpZerosLike);

#undef REG_TFL_OP
  }

private:
  std::map<tflite::BuiltinOperator, std::unique_ptr<TFliteOpChef>> _tfliteop_map;
};

} // namespace tflchef

#endif // __TFLITE_OP_REGISTRY_H__
