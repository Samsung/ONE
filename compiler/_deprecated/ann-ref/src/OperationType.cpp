/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright (C) 2017 The Android Open Source Project
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

#include "OperationType.h"
#include "Macro.h"

const char *kOperationNames[kNumberOfOperationTypes] = {
    "ADD",
    "AVERAGE_POOL",
    "CONCATENATION",
    "CONV",
    "DEPTHWISE_CONV",
    "DEPTH_TO_SPACE",
    "DEQUANTIZE",
    "EMBEDDING_LOOKUP",
    "FLOOR",
    "FULLY_CONNECTED",
    "HASHTABLE_LOOKUP",
    "L2_NORMALIZATION",
    "L2_POOL",
    "LOCAL_RESPONSE_NORMALIZATION",
    "LOGISTIC",
    "LSH_PROJECTION",
    "LSTM",
    "MAX_POOL",
    "MUL",
    "RELU",
    "RELU1",
    "RELU6",
    "RESHAPE",
    "RESIZE_BILINEAR",
    "RNN",
    "SOFTMAX",
    "SPACE_TO_DEPTH",
    "SVDF",
    "TANH",
    "BATCH_TO_SPACE_ND", // V1_1, will not be merged till V1_1 is finalized
    "DIV",
    "MEAN",              // V1_1, will not be merged till V1_1 is finalized
    "PAD",               // V1_1, will not be merged till V1_1 is finalized
    "SPACE_TO_BATCH_ND", // V1_1, will not be merged till V1_1 is finalized
    "SQUEEZE",           // V1_1, will not be merged till V1_1 is finalized
    "STRIDED_SLICE",
    "SUB",
};

static_assert(COUNT(kOperationNames) == kNumberOfOperationTypes, "kOperationNames is incorrect");

const char *getOperationName(OperationType type)
{
  uint32_t n = static_cast<uint32_t>(type);
  return kOperationNames[n];
}
