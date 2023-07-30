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

  CIRCLE_NODE(ABS, CircleAbsGraphBuilder);                                                 // 101
  CIRCLE_NODE(ADD, CircleAddGraphBuilder);                                                 // 0
  CIRCLE_NODE(ADD_N, CircleAddNGraphBuilder);                                              // 106
  CIRCLE_NODE(ARG_MAX, CircleArgMaxGraphBuilder);                                          // 56
  CIRCLE_NODE(ARG_MIN, CircleArgMinGraphBuilder);                                          // 79
  CIRCLE_NODE(AVERAGE_POOL_2D, CircleAveragePool2DGraphBuilder);                           // 1
  CIRCLE_NODE(BATCH_MATMUL, CircleBatchMatMulGraphBuilder);                                // 126
  CIRCLE_NODE(BATCH_TO_SPACE_ND, CircleBatchToSpaceNDGraphBuilder);                        // 37
  CIRCLE_NODE(BCQ_FULLY_CONNECTED, CircleBCQFullyConnectedGraphBuilder);                   // 253
  CIRCLE_NODE(BCQ_GATHER, CircleBCQGatherGraphBuilder);                                    // 252
  CIRCLE_NODE(BIDIRECTIONAL_SEQUENCE_LSTM, CircleBidirectionalSequenceLSTMGraphBuilder);   // 52
  CIRCLE_NODE(CAST, CircleCastGraphBuilder);                                               // 53
  CIRCLE_NODE(CEIL, CircleCeilGraphBuilder);                                               // 104
  CIRCLE_NODE(CUSTOM, CircleCustomGraphBuilder);                                           // 32
  CIRCLE_NODE(CONCATENATION, CircleConcatenationGraphBuilder);                             // 2
  CIRCLE_NODE(CONV_2D, CircleConv2DGraphBuilder);                                          // 3
  CIRCLE_NODE(COS, CircleCosGraphBuilder);                                                 // 108
  CIRCLE_NODE(DENSIFY, CircleDensifyGraphBuilder);                                         // 124
  CIRCLE_NODE(DEPTH_TO_SPACE, CircleDepthToSpaceGraphBuilder);                             // 5
  CIRCLE_NODE(DEPTHWISE_CONV_2D, CircleDepthwiseConv2DGraphBuilder);                       // 4
  CIRCLE_NODE(DEQUANTIZE, CircleDequantizeGraphBuilder);                                   // 6
  CIRCLE_NODE(DIV, CircleDivGraphBuilder);                                                 // 42
  CIRCLE_NODE(ELU, CircleEluGraphBuilder);                                                 // 111
  CIRCLE_NODE(EQUAL, CircleEqualGraphBuilder);                                             // 71
  CIRCLE_NODE(EXP, CircleExpGraphBuilder);                                                 // 47
  CIRCLE_NODE(EXPAND_DIMS, CircleExpandDimsGraphBuilder);                                  // 70
  CIRCLE_NODE(FAKE_QUANT, CircleFakeQuantGraphBuilder);                                    // 80
  CIRCLE_NODE(FILL, CircleFillGraphBuilder);                                               // 94
  CIRCLE_NODE(FLOOR, CircleFloorGraphBuilder);                                             // 8
  CIRCLE_NODE(FLOOR_DIV, CircleFloorDivGraphBuilder);                                      // 90
  CIRCLE_NODE(FLOOR_MOD, CircleFloorModGraphBuilder);                                      // 95
  CIRCLE_NODE(FULLY_CONNECTED, CircleFullyConnectedGraphBuilder);                          // 9
  CIRCLE_NODE(GATHER, CircleGatherGraphBuilder);                                           // 36
  CIRCLE_NODE(GATHER_ND, CircleGatherNdGraphBuilder);                                      // 107
  CIRCLE_NODE(GELU, CircleGeluGraphBuilder);                                               // 150
  CIRCLE_NODE(GREATER, CircleGreaterGraphBuilder);                                         // 61
  CIRCLE_NODE(GREATER_EQUAL, CircleGreaterEqualGraphBuilder);                              // 62
  CIRCLE_NODE(HARD_SWISH, CircleHardSwishGraphBuilder);                                    // 117
  CIRCLE_NODE(IF, CircleIfGraphBuilder);                                                   // 118
  CIRCLE_NODE(INSTANCE_NORM, CircleInstanceNormGraphBuilder);                              // 254
  CIRCLE_NODE(L2_NORMALIZATION, CircleL2NormalizeGraphBuilder);                            // 11
  CIRCLE_NODE(L2_POOL_2D, CircleL2Pool2DGraphBuilder);                                     // 12
  CIRCLE_NODE(LEAKY_RELU, CircleLeakyReluGraphBuilder);                                    // 98,
  CIRCLE_NODE(LESS, CircleLessGraphBuilder);                                               // 58
  CIRCLE_NODE(LESS_EQUAL, CircleLessEqualGraphBuilder);                                    // 63
  CIRCLE_NODE(LOCAL_RESPONSE_NORMALIZATION, CircleLocalResponseNormalizationGraphBuilder); // 13
  CIRCLE_NODE(LOG, CircleLogGraphBuilder);                                                 // 73
  CIRCLE_NODE(LOGICAL_AND, CircleLogicalAndGraphBuilder);                                  // 86
  CIRCLE_NODE(LOGICAL_NOT, CircleLogicalNotGraphBuilder);                                  // 87
  CIRCLE_NODE(LOGICAL_OR, CircleLogicalOrGraphBuilder);                                    // 84
  CIRCLE_NODE(LOGISTIC, CircleLogisticGraphBuilder);                                       // 14
  CIRCLE_NODE(LOG_SOFTMAX, CircleLogSoftmaxGraphBuilder);                                  // 50
  CIRCLE_NODE(MATRIX_DIAG, CircleMatrixDiagGraphBuilder);                                  // 113
  CIRCLE_NODE(MATRIX_SET_DIAG, CircleMatrixSetDiagGraphBuilder);                           // 115
  CIRCLE_NODE(MAXIMUM, CircleMaximumGraphBuilder);                                         // 55
  CIRCLE_NODE(MAX_POOL_2D, CircleMaxPool2DGraphBuilder);                                   // 17
  CIRCLE_NODE(MEAN, CircleMeanGraphBuilder);                                               // 40
  CIRCLE_NODE(MINIMUM, CircleMinimumGraphBuilder);                                         // 57
  CIRCLE_NODE(MIRROR_PAD, CircleMirrorPadGraphBuilder);                                    // 100
  CIRCLE_NODE(MUL, CircleMulGraphBuilder);                                                 // 18
  CIRCLE_NODE(NEG, CircleNegGraphBuilder);                                                 // 59
  CIRCLE_NODE(NON_MAX_SUPPRESSION_V4, CircleNonMaxSuppressionV4GraphBuilder);              // 120,
  CIRCLE_NODE(NON_MAX_SUPPRESSION_V5, CircleNonMaxSuppressionV5GraphBuilder);              // 121,
  CIRCLE_NODE(NOT_EQUAL, CircleNotEqualGraphBuilder);                                      // 72
  CIRCLE_NODE(ONE_HOT, CircleOneHotGraphBuilder);                                          // 85
  CIRCLE_NODE(PACK, CirclePackGraphBuilder);                                               // 83
  CIRCLE_NODE(PAD, CirclePadGraphBuilder);                                                 // 34
  CIRCLE_NODE(PADV2, CirclePadV2GraphBuilder);                                             // 60
  CIRCLE_NODE(POW, CirclePowGraphBuilder);                                                 // 78
  CIRCLE_NODE(PRELU, CirclePReluGraphBuilder);                                             // 54,
  CIRCLE_NODE(QUANTIZE, CircleQuantizeGraphBuilder);                                       // 114,
  CIRCLE_NODE(RANGE, CircleRangeGraphBuilder);                                             // 96
  CIRCLE_NODE(RANK, CircleRankGraphBuilder);                                               // 110
  CIRCLE_NODE(REDUCE_ANY, CircleReduceAnyGraphBuilder);                                    // 91
  CIRCLE_NODE(REDUCE_MAX, CircleReduceMaxGraphBuilder);                                    // 82
  CIRCLE_NODE(REDUCE_MIN, CircleReduceMinGraphBuilder);                                    // 89
  CIRCLE_NODE(REDUCE_PROD, CircleReduceProdGraphBuilder);                                  // 81
  CIRCLE_NODE(RELU, CircleReluGraphBuilder);                                               // 19
  CIRCLE_NODE(RELU6, CircleRelu6GraphBuilder);                                             // 21
  CIRCLE_NODE(RELU_N1_TO_1, CircleReluN1To1GraphBuilder);                                  // 20
  CIRCLE_NODE(RESHAPE, CircleReshapeGraphBuilder);                                         // 22
  CIRCLE_NODE(RESIZE_BILINEAR, CircleResizeBilinearGraphBuilder);                          // 23
  CIRCLE_NODE(RESIZE_NEAREST_NEIGHBOR, CircleResizeNearestNeighborGraphBuilder);           // 97
  CIRCLE_NODE(REVERSE_SEQUENCE, CircleReverseSequenceGraphBuilder);                        // 112
  CIRCLE_NODE(REVERSE_V2, CircleReverseV2GraphBuilder);                                    // 105
  CIRCLE_NODE(ROUND, CircleRoundGraphBuilder);                                             // 116
  CIRCLE_NODE(RSQRT, CircleRsqrtGraphBuilder);                                             // 76
  CIRCLE_NODE(SCATTER_ND, CircleScatterNdGraphBuilder);                                    // 122
  CIRCLE_NODE(SEGMENT_SUM, CircleSegmentSumGraphBuilder);                                  // 125
  CIRCLE_NODE(SELECT, CircleSelectGraphBuilder);                                           // 64
  CIRCLE_NODE(SELECT_V2, CircleSelectV2GraphBuilder);                                      // 123
  CIRCLE_NODE(SHAPE, CircleShapeGraphBuilder);                                             // 77
  CIRCLE_NODE(SIN, CircleSinGraphBuilder);                                                 // 66
  CIRCLE_NODE(SLICE, CircleSliceGraphBuilder);                                             // 65
  CIRCLE_NODE(SOFTMAX, CircleSoftmaxGraphBuilder);                                         // 25
  CIRCLE_NODE(SPACE_TO_BATCH_ND, CircleSpaceToBatchNDGraphBuilder);                        // 38
  CIRCLE_NODE(SPACE_TO_DEPTH, CircleSpaceToDepthGraphBuilder);                             // 26
  CIRCLE_NODE(SPARSE_TO_DENSE, CircleSparseToDenseGraphBuilder);                           // 68
  CIRCLE_NODE(SPLIT, CircleSplitGraphBuilder);                                             // 49
  CIRCLE_NODE(SPLIT_V, CircleSplitVGraphBuilder);                                          // 102
  CIRCLE_NODE(SQRT, CircleSqrtGraphBuilder);                                               // 75
  CIRCLE_NODE(SQUARE, CircleSquareGraphBuilder);                                           // 92
  CIRCLE_NODE(SQUARED_DIFFERENCE, CircleSquaredDifferenceGraphBuilder);                    // 99
  CIRCLE_NODE(SQUEEZE, CircleSqueezeGraphBuilder);                                         // 43
  CIRCLE_NODE(STRIDED_SLICE, CircleStridedSliceGraphBuilder);                              // 45
  CIRCLE_NODE(SUB, CircleSubGraphBuilder);                                                 // 41
  CIRCLE_NODE(SUM, CircleSumGraphBuilder);                                                 // 74
  CIRCLE_NODE(SVDF, CircleSVDFBuilder);                                                    // 27
  CIRCLE_NODE(TANH, CircleTanhGraphBuilder);                                               // 28
  CIRCLE_NODE(TILE, CircleTileGraphBuilder);                                               // 69
  CIRCLE_NODE(TOPK_V2, CircleTopKV2GraphBuilder);                                          // 48
  CIRCLE_NODE(TRANSPOSE, CircleTransposeGraphBuilder);                                     // 39
  CIRCLE_NODE(TRANSPOSE_CONV, CircleTransposeConvGraphBuilder);                            // 67
  CIRCLE_NODE(UNIDIRECTIONAL_SEQUENCE_LSTM, CircleUnidirectionalSequenceLSTMGraphBuilder); // 44
  CIRCLE_NODE(UNIQUE, CircleUniqueGraphBuilder);                                           // 103
  CIRCLE_NODE(UNPACK, CircleUnpackGraphBuilder);                                           // 88
  CIRCLE_NODE(WHERE, CircleWhereGraphBuilder);                                             // 109
  CIRCLE_NODE(WHILE, CircleWhileGraphBuilder);                                             // 119
  CIRCLE_NODE(ZEROS_LIKE, CircleZerosLikeGraphBuilder);                                    // 93

#undef CIRCLE_NODE

  // BuiltinOperator_EMBEDDING_LOOKUP = 7,
  // BuiltinOperator_HASHTABLE_LOOKUP = 10,
  // BuiltinOperator_LSH_PROJECTION = 15,
  // BuiltinOperator_LSTM = 16,
  // BuiltinOperator_RNN = 24,
  // BuiltinOperator_CONCAT_EMBEDDINGS = 29,
  // BuiltinOperator_SKIP_GRAM = 30,
  // BuiltinOperator_CALL = 31,
  // BuiltinOperator_EMBEDDING_LOOKUP_SPARSE = 33,
  // BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_RNN = 35,
  // BuiltinOperator_BIDIRECTIONAL_SEQUENCE_RNN = 46,
  // BuiltinOperator_DELEGATE = 51,
  // BuiltinOperator_ARG_MAX = 56,
  // BuiltinOperator_HARD_SWISH = 117,

  // Register builders for nodes which not handles in builders registered above.
#define CIRCLE_NODE(CLASS) add(std::make_unique<CLASS>())

  CIRCLE_NODE(CircleConstNodeBuilder);

#undef CIRCLE_NODE
}

} // namespace luci
