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

#include "NeuralNetworks.h"

// Make sure the constants defined in the header files have not changed values.
// IMPORTANT: When adding new values, update kNumberOfDataTypes or kNumberOfDataTypesOEM
// in Utils.h.
static_assert(ANEURALNETWORKS_FLOAT32 == 0, "ANEURALNETWORKS_FLOAT32 has changed");
static_assert(ANEURALNETWORKS_INT32 == 1, "ANEURALNETWORKS_INT32 has changed");
static_assert(ANEURALNETWORKS_UINT32 == 2, "ANEURALNETWORKS_UINT32 has changed");
static_assert(ANEURALNETWORKS_TENSOR_FLOAT32 == 3, "ANEURALNETWORKS_TENSOR_FLOAT32 has changed");
static_assert(ANEURALNETWORKS_TENSOR_INT32 == 4, "ANEURALNETWORKS_TENSOR_INT32 has changed");
static_assert(ANEURALNETWORKS_TENSOR_QUANT8_ASYMM == 5,
              "ANEURALNETWORKS_TENSOR_QUANT8_ASYMM has changed");

// IMPORTANT: When adding new values, update kNumberOfOperationTypes or
// kNumberOfOperationTypesOEM kNumberOfOperationTypesEx in Utils.h.
static_assert(ANEURALNETWORKS_ADD == 0, "ANEURALNETWORKS_ADD has changed");
static_assert(ANEURALNETWORKS_AVERAGE_POOL_2D == 1, "ANEURALNETWORKS_AVERAGE_POOL_2D has changed");
static_assert(ANEURALNETWORKS_CONCATENATION == 2, "ANEURALNETWORKS_CONCATENATION has changed");
static_assert(ANEURALNETWORKS_CONV_2D == 3, "ANEURALNETWORKS_CONV_2D has changed");
static_assert(ANEURALNETWORKS_DEPTHWISE_CONV_2D == 4,
              "ANEURALNETWORKS_DEPTHWISE_CONV_2D has changed");
static_assert(ANEURALNETWORKS_DEPTH_TO_SPACE == 5, "ANEURALNETWORKS_DEPTH_TO_SPACE has changed");
static_assert(ANEURALNETWORKS_DEQUANTIZE == 6, "ANEURALNETWORKS_DEQUANTIZE has changed");
static_assert(ANEURALNETWORKS_EMBEDDING_LOOKUP == 7,
              "ANEURALNETWORKS_EMBEDDING_LOOKUP has changed");
static_assert(ANEURALNETWORKS_FLOOR == 8, "ANEURALNETWORKS_FLOOR has changed");
static_assert(ANEURALNETWORKS_FULLY_CONNECTED == 9, "ANEURALNETWORKS_FULLY_CONNECTED has changed");
static_assert(ANEURALNETWORKS_HASHTABLE_LOOKUP == 10,
              "ANEURALNETWORKS_HASHTABLE_LOOKUP has changed");
static_assert(ANEURALNETWORKS_L2_NORMALIZATION == 11,
              "ANEURALNETWORKS_L2_NORMALIZATION has changed");
static_assert(ANEURALNETWORKS_L2_POOL_2D == 12, "ANEURALNETWORKS_L2_POOL has changed");
static_assert(ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION == 13,
              "ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION has changed");
static_assert(ANEURALNETWORKS_LOGISTIC == 14, "ANEURALNETWORKS_LOGISTIC has changed");
static_assert(ANEURALNETWORKS_LSH_PROJECTION == 15, "ANEURALNETWORKS_LSH_PROJECTION has changed");
static_assert(ANEURALNETWORKS_LSTM == 16, "ANEURALNETWORKS_LSTM has changed");
static_assert(ANEURALNETWORKS_MAX_POOL_2D == 17, "ANEURALNETWORKS_MAX_POOL has changed");
static_assert(ANEURALNETWORKS_MUL == 18, "ANEURALNETWORKS_MUL has changed");
static_assert(ANEURALNETWORKS_RELU == 19, "ANEURALNETWORKS_RELU has changed");
static_assert(ANEURALNETWORKS_RELU1 == 20, "ANEURALNETWORKS_RELU1 has changed");
static_assert(ANEURALNETWORKS_RELU6 == 21, "ANEURALNETWORKS_RELU6 has changed");
static_assert(ANEURALNETWORKS_RESHAPE == 22, "ANEURALNETWORKS_RESHAPE has changed");
static_assert(ANEURALNETWORKS_RESIZE_BILINEAR == 23, "ANEURALNETWORKS_RESIZE_BILINEAR has changed");
static_assert(ANEURALNETWORKS_RNN == 24, "ANEURALNETWORKS_RNN has changed");
static_assert(ANEURALNETWORKS_SOFTMAX == 25, "ANEURALNETWORKS_SOFTMAX has changed");
static_assert(ANEURALNETWORKS_SPACE_TO_DEPTH == 26, "ANEURALNETWORKS_SPACE_TO_DEPTH has changed");
static_assert(ANEURALNETWORKS_SVDF == 27, "ANEURALNETWORKS_SVDF has changed");
static_assert(ANEURALNETWORKS_TANH == 28, "ANEURALNETWORKS_TANH has changed");

static_assert(ANEURALNETWORKS_FUSED_NONE == 0, "ANEURALNETWORKS_FUSED_NONE has changed");
static_assert(ANEURALNETWORKS_FUSED_RELU == 1, "ANEURALNETWORKS_FUSED_RELU has changed");
static_assert(ANEURALNETWORKS_FUSED_RELU1 == 2, "ANEURALNETWORKS_FUSED_RELU1 has changed");
static_assert(ANEURALNETWORKS_FUSED_RELU6 == 3, "ANEURALNETWORKS_FUSED_RELU6 has changed");

static_assert(ANEURALNETWORKS_PREFER_LOW_POWER == 0,
              "ANEURALNETWORKS_PREFER_LOW_POWER has changed");
static_assert(ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER == 1,
              "ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER has changed");
static_assert(ANEURALNETWORKS_PREFER_SUSTAINED_SPEED == 2,
              "ANEURALNETWORKS_PREFER_SUSTAINED_SPEED has changed");

static_assert(ANEURALNETWORKS_NO_ERROR == 0, "ANEURALNETWORKS_NO_ERROR has changed");
static_assert(ANEURALNETWORKS_OUT_OF_MEMORY == 1, "ANEURALNETWORKS_OUT_OF_MEMORY has changed");
static_assert(ANEURALNETWORKS_INCOMPLETE == 2, "ANEURALNETWORKS_INCOMPLETE has changed");
static_assert(ANEURALNETWORKS_UNEXPECTED_NULL == 3, "ANEURALNETWORKS_UNEXPECTED_NULL has changed");
static_assert(ANEURALNETWORKS_BAD_DATA == 4, "ANEURALNETWORKS_BAD_DATA has changed");
static_assert(ANEURALNETWORKS_OP_FAILED == 5, "ANEURALNETWORKS_OP_FAILED has changed");
static_assert(ANEURALNETWORKS_BAD_STATE == 6, "ANEURALNETWORKS_BAD_STATE has changed");

static_assert(ANEURALNETWORKS_MAX_SIZE_OF_IMMEDIATELY_COPIED_VALUES == 128,
              "ANEURALNETWORKS_MAX_SIZE_OF_IMMEDIATELY_COPIED_VALUES has changed");
