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

#include <stdex/Memory.h>

namespace luci
{

GraphBuilderRegistry::GraphBuilderRegistry()
{
  add(circle::BuiltinOperator_ADD, stdex::make_unique<CircleAddGraphBuilder>());               // 0
  add(circle::BuiltinOperator_ARG_MAX, stdex::make_unique<CircleArgMaxGraphBuilder>());        // 56
  add(circle::BuiltinOperator_CONV_2D, stdex::make_unique<CircleConv2DGraphBuilder>());        // 3
  add(circle::BuiltinOperator_MAX_POOL_2D, stdex::make_unique<CircleMaxPool2DGraphBuilder>()); // 17
  add(circle::BuiltinOperator_MEAN, stdex::make_unique<CircleMeanGraphBuilder>());             // 40
  add(circle::BuiltinOperator_PAD, stdex::make_unique<CirclePadGraphBuilder>());               // 34
  add(circle::BuiltinOperator_RESHAPE, stdex::make_unique<CircleReshapeGraphBuilder>());       // 22

  // BuiltinOperator_AVERAGE_POOL_2D = 1,
  // BuiltinOperator_CONCATENATION = 2,
  // BuiltinOperator_DEPTHWISE_CONV_2D = 4,
  // BuiltinOperator_DEQUANTIZE = 6,
  // BuiltinOperator_EMBEDDING_LOOKUP = 7,
  // BuiltinOperator_FLOOR = 8,
  // BuiltinOperator_FULLY_CONNECTED = 9,
  // BuiltinOperator_HASHTABLE_LOOKUP = 10,
  // BuiltinOperator_L2_NORMALIZATION = 11,
  // BuiltinOperator_L2_POOL_2D = 12,
  // BuiltinOperator_LOCAL_RESPONSE_NORMALIZATION = 13,
  // BuiltinOperator_LOGISTIC = 14,
  // BuiltinOperator_LSH_PROJECTION = 15,
  // BuiltinOperator_LSTM = 16,
  // BuiltinOperator_MUL = 18,
  // BuiltinOperator_RELU = 19,
  // BuiltinOperator_RELU_N1_TO_1 = 20,
  // BuiltinOperator_RELU6 = 21,
  // BuiltinOperator_RESIZE_BILINEAR = 23,
  // BuiltinOperator_RNN = 24,
  // BuiltinOperator_SOFTMAX = 25,
  // BuiltinOperator_SPACE_TO_DEPTH = 26,
  // BuiltinOperator_SVDF = 27,
  // BuiltinOperator_TANH = 28,
  // BuiltinOperator_CONCAT_EMBEDDINGS = 29,
  // BuiltinOperator_SKIP_GRAM = 30,
  // BuiltinOperator_CALL = 31,
  // BuiltinOperator_CUSTOM = 32,
  // BuiltinOperator_EMBEDDING_LOOKUP_SPARSE = 33,
  // BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_RNN = 35,
  // BuiltinOperator_GATHER = 36,
  // BuiltinOperator_BATCH_TO_SPACE_ND = 37,
  // BuiltinOperator_SPACE_TO_BATCH_ND = 38,
  // BuiltinOperator_TRANSPOSE = 39,
  // BuiltinOperator_SUB = 41,
  // BuiltinOperator_DIV = 42,
  // BuiltinOperator_SQUEEZE = 43,
  // BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM = 44,
  // BuiltinOperator_STRIDED_SLICE = 45,
  // BuiltinOperator_BIDIRECTIONAL_SEQUENCE_RNN = 46,
  // BuiltinOperator_EXP = 47,
  // BuiltinOperator_TOPK_V2 = 48,
  // BuiltinOperator_SPLIT = 49,
  // BuiltinOperator_LOG_SOFTMAX = 50,
  // BuiltinOperator_DELEGATE = 51,
  // BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM = 52,
  // BuiltinOperator_CAST = 53,
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
  // BuiltinOperator_SIN = 66,
  // BuiltinOperator_TRANSPOSE_CONV = 67,
  // BuiltinOperator_SPARSE_TO_DENSE = 68,
  // BuiltinOperator_TILE = 69,
  // BuiltinOperator_EXPAND_DIMS = 70,
  // BuiltinOperator_EQUAL = 71,
  // BuiltinOperator_NOT_EQUAL = 72,
  // BuiltinOperator_LOG = 73,
  // BuiltinOperator_SUM = 74,
  // BuiltinOperator_SQRT = 75,
  // BuiltinOperator_RSQRT = 76,
  // BuiltinOperator_SHAPE = 77,
  // BuiltinOperator_POW = 78,
  // BuiltinOperator_ARG_MIN = 79,
  // BuiltinOperator_FAKE_QUANT = 80,
  // BuiltinOperator_REDUCE_PROD = 81,
  // BuiltinOperator_REDUCE_MAX = 82,
  // BuiltinOperator_PACK = 83,
  // BuiltinOperator_LOGICAL_OR = 84,
  // BuiltinOperator_ONE_HOT = 85,
  // BuiltinOperator_LOGICAL_AND = 86,
  // BuiltinOperator_LOGICAL_NOT = 87,
  // BuiltinOperator_UNPACK = 88,
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
  // BuiltinOperator_SQUARED_DIFFERENCE = 99,
  // BuiltinOperator_MIRROR_PAD = 100,
  // BuiltinOperator_ABS = 101,
  // BuiltinOperator_SPLIT_V = 102,
  // BuiltinOperator_INSTANCE_NORM = 254,
}

} // namespace luci
