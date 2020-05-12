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

#include "luci/Import/GraphBuilderRegistry.h"

#include "luci/Import/Nodes.h"

#include <memory>

namespace luci
{

GraphBuilderRegistry::GraphBuilderRegistry()
{
#define CIRCLE_NODE(OPCODE, CLASS) add(circle::BuiltinOperator_##OPCODE, std::make_unique<CLASS>());

  CIRCLE_NODE(ABS, CircleAbsGraphBuilder);                              // 101
  CIRCLE_NODE(ADD, CircleAddGraphBuilder);                              // 0
  CIRCLE_NODE(ARG_MAX, CircleArgMaxGraphBuilder);                       // 56
  CIRCLE_NODE(AVERAGE_POOL_2D, CircleAveragePool2DGraphBuilder);        // 1
  CIRCLE_NODE(BATCH_TO_SPACE_ND, CircleBatchToSpaceNDGraphBuilder);     // 37
  CIRCLE_NODE(CAST, CircleCastGraphBuilder);                            // 53
  CIRCLE_NODE(CUSTOM, CircleCustomGraphBuilder);                        // 32
  CIRCLE_NODE(CONCATENATION, CircleConcatenationGraphBuilder);          // 2
  CIRCLE_NODE(CONV_2D, CircleConv2DGraphBuilder);                       // 3
  CIRCLE_NODE(COS, CircleCosGraphBuilder);                              // 108
  CIRCLE_NODE(DEPTHWISE_CONV_2D, CircleDepthwiseConv2DGraphBuilder);    // 4
  CIRCLE_NODE(DIV, CircleDivGraphBuilder);                              // 42
  CIRCLE_NODE(EQUAL, CircleEqualGraphBuilder);                          // 71
  CIRCLE_NODE(EXP, CircleExpGraphBuilder);                              // 47
  CIRCLE_NODE(FULLY_CONNECTED, CircleFullyConnectedGraphBuilder);       // 9
  CIRCLE_NODE(GATHER, CircleGatherGraphBuilder);                        // 36
  CIRCLE_NODE(IF, CircleIfGraphBuilder);                                // 118
  CIRCLE_NODE(LOGICAL_NOT, CircleLogicalNotGraphBuilder);               // 87
  CIRCLE_NODE(LOGICAL_OR, CircleLogicalOrGraphBuilder);                 // 84
  CIRCLE_NODE(LOGISTIC, CircleLogisticGraphBuilder);                    // 14
  CIRCLE_NODE(MAX_POOL_2D, CircleMaxPool2DGraphBuilder);                // 17
  CIRCLE_NODE(MEAN, CircleMeanGraphBuilder);                            // 40
  CIRCLE_NODE(MUL, CircleMulGraphBuilder);                              // 18
  CIRCLE_NODE(PACK, CirclePackGraphBuilder);                            // 83
  CIRCLE_NODE(PAD, CirclePadGraphBuilder);                              // 34
  CIRCLE_NODE(REDUCE_PROD, CircleReduceProdGraphBuilder);               // 81
  CIRCLE_NODE(RELU, CircleReluGraphBuilder);                            // 19
  CIRCLE_NODE(RESHAPE, CircleReshapeGraphBuilder);                      // 22
  CIRCLE_NODE(RSQRT, CircleRsqrtGraphBuilder);                          // 76
  CIRCLE_NODE(SIN, CircleSinGraphBuilder);                              // 66
  CIRCLE_NODE(SOFTMAX, CircleSoftmaxGraphBuilder);                      // 25
  CIRCLE_NODE(SPACE_TO_BATCH_ND, CircleSpaceToBatchNDGraphBuilder);     // 38
  CIRCLE_NODE(SPLIT, CircleSplitGraphBuilder);                          // 49
  CIRCLE_NODE(SPLIT_V, CircleSplitVGraphBuilder);                       // 102
  CIRCLE_NODE(SQUARED_DIFFERENCE, CircleSquaredDifferenceGraphBuilder); // 99
  CIRCLE_NODE(STRIDED_SLICE, CircleStridedSliceGraphBuilder);           // 45
  CIRCLE_NODE(SUB, CircleSubGraphBuilder);                              // 41
  CIRCLE_NODE(TANH, CircleTanhGraphBuilder);                            // 28
  CIRCLE_NODE(TILE, CircleTileGraphBuilder);                            // 69
  CIRCLE_NODE(TRANSPOSE, CircleTransposeGraphBuilder);                  // 39
  CIRCLE_NODE(UNPACK, CircleUnpackGraphBuilder);                        // 88
  CIRCLE_NODE(WHILE, CircleWhileGraphBuilder);                          // 119

#undef CIRCLE_NODE

  // BuiltinOperator_DEQUANTIZE = 6,
  // BuiltinOperator_EMBEDDING_LOOKUP = 7,
  // BuiltinOperator_FLOOR = 8,
  // BuiltinOperator_HASHTABLE_LOOKUP = 10,
  // BuiltinOperator_L2_NORMALIZATION = 11,
  // BuiltinOperator_L2_POOL_2D = 12,
  // BuiltinOperator_LOCAL_RESPONSE_NORMALIZATION = 13,
  // BuiltinOperator_LSH_PROJECTION = 15,
  // BuiltinOperator_LSTM = 16,
  // BuiltinOperator_RELU_N1_TO_1 = 20,
  // BuiltinOperator_RELU6 = 21,
  // BuiltinOperator_RESIZE_BILINEAR = 23,
  // BuiltinOperator_RNN = 24,
  // BuiltinOperator_SPACE_TO_DEPTH = 26,
  // BuiltinOperator_SVDF = 27,
  // BuiltinOperator_CONCAT_EMBEDDINGS = 29,
  // BuiltinOperator_SKIP_GRAM = 30,
  // BuiltinOperator_CALL = 31,
  // BuiltinOperator_EMBEDDING_LOOKUP_SPARSE = 33,
  // BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_RNN = 35,
  // BuiltinOperator_SQUEEZE = 43,
  // BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM = 44,
  // BuiltinOperator_BIDIRECTIONAL_SEQUENCE_RNN = 46,
  // BuiltinOperator_TOPK_V2 = 48,
  // BuiltinOperator_LOG_SOFTMAX = 50,
  // BuiltinOperator_DELEGATE = 51,
  // BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM = 52,
  // BuiltinOperator_PRELU = 54,
  // BuiltinOperator_MAXIMUM = 55,
  // BuiltinOperator_ARG_MAX = 56,
  // BuiltinOperator_MINIMUM = 57,
  // BuiltinOperator_LESS = 58,
  // BuiltinOperator_NEG = 59,
  // BuiltinOperator_PADV2 = 60,
  // BuiltinOperator_GREATER = 61,
  // BuiltinOperator_GREATER_EQUAL = 62,
  // BuiltinOperator_LESS_EQUAL = 63,
  // BuiltinOperator_SELECT = 64,
  // BuiltinOperator_SLICE = 65,
  // BuiltinOperator_TRANSPOSE_CONV = 67,
  // BuiltinOperator_SPARSE_TO_DENSE = 68,
  // BuiltinOperator_EXPAND_DIMS = 70,
  // BuiltinOperator_NOT_EQUAL = 72,
  // BuiltinOperator_LOG = 73,
  // BuiltinOperator_SUM = 74,
  // BuiltinOperator_SQRT = 75,
  // BuiltinOperator_SHAPE = 77,
  // BuiltinOperator_POW = 78,
  // BuiltinOperator_ARG_MIN = 79,
  // BuiltinOperator_FAKE_QUANT = 80,
  // BuiltinOperator_REDUCE_MAX = 82,
  // BuiltinOperator_ONE_HOT = 85,
  // BuiltinOperator_LOGICAL_AND = 86,
  // BuiltinOperator_REDUCE_MIN = 89,
  // BuiltinOperator_FLOOR_DIV = 90,
  // BuiltinOperator_REDUCE_ANY = 91,
  // BuiltinOperator_SQUARE = 92,
  // BuiltinOperator_ZEROS_LIKE = 93,
  // BuiltinOperator_FILL = 94,
  // BuiltinOperator_FLOOR_MOD = 95,
  // BuiltinOperator_RANGE = 96,
  // BuiltinOperator_RESIZE_NEAREST_NEIGHBOR = 97,
  // BuiltinOperator_LEAKY_RELU = 98,
  // BuiltinOperator_MIRROR_PAD = 100,
  // BuiltinOperator_UNIQUE = 103,
  // BuiltinOperator_CEIL = 104,
  // BuiltinOperator_REVERSE_V2 = 105,
  // BuiltinOperator_ADD_N = 106,
  // BuiltinOperator_GATHER_ND = 107,
  // BuiltinOperator_WHERE = 109,
  // BuiltinOperator_RANK = 110,
  // BuiltinOperator_ELU = 111,
  // BuiltinOperator_REVERSE_SEQUENCE = 112,
  // BuiltinOperator_MATRIX_DIAG = 113,
  // BuiltinOperator_QUANTIZE = 114,
  // BuiltinOperator_MATRIX_SET_DIAG = 115,
  // BuiltinOperator_ROUND = 116,
  // BuiltinOperator_HARD_SWISH = 117,
  // BuiltinOperator_NON_MAX_SUPPRESSION_V4 = 120,
  // BuiltinOperator_NON_MAX_SUPPRESSION_V5 = 121,
  // BuiltinOperator_SCATTER_ND = 122,
  // BuiltinOperator_SELECT_V2 = 123,
  // BuiltinOperator_DENSIFY = 124,
  // BuiltinOperator_SEGMENT_SUM = 125,
  // BuiltinOperator_BATCH_MATMUL = 126,
  // BuiltinOperator_INSTANCE_NORM = 254,
}

} // namespace luci
