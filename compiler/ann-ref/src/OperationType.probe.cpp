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
#include "NeuralNetworks.h"

static_assert(static_cast<int32_t>(OperationType::ADD) == ANEURALNETWORKS_ADD,
              "OperationType::ADD != ANEURALNETWORKS_ADD");
static_assert(static_cast<int32_t>(OperationType::AVERAGE_POOL_2D) ==
                  ANEURALNETWORKS_AVERAGE_POOL_2D,
              "OperationType::AVERAGE_POOL_2D != ANEURALNETWORKS_AVERAGE_POOL_2D");
static_assert(static_cast<int32_t>(OperationType::CONV_2D) == ANEURALNETWORKS_CONV_2D,
              "OperationType::CONV_2D != ANEURALNETWORKS_CONV_2D");
static_assert(static_cast<int32_t>(OperationType::DEPTHWISE_CONV_2D) ==
                  ANEURALNETWORKS_DEPTHWISE_CONV_2D,
              "OperationType::DEPTHWISE_CONV_2D != ANEURALNETWORKS_DEPTHWISE_CONV_2D");
static_assert(static_cast<int32_t>(OperationType::DEPTH_TO_SPACE) == ANEURALNETWORKS_DEPTH_TO_SPACE,
              "OperationType::DEPTH_TO_SPACE != ANEURALNETWORKS_DEPTH_TO_SPACE");
static_assert(static_cast<int32_t>(OperationType::DEQUANTIZE) == ANEURALNETWORKS_DEQUANTIZE,
              "OperationType::DEQUANTIZE != ANEURALNETWORKS_DEQUANTIZE");
static_assert(static_cast<int32_t>(OperationType::EMBEDDING_LOOKUP) ==
                  ANEURALNETWORKS_EMBEDDING_LOOKUP,
              "OperationType::EMBEDDING_LOOKUP != ANEURALNETWORKS_EMBEDDING_LOOKUP");
static_assert(static_cast<int32_t>(OperationType::FLOOR) == ANEURALNETWORKS_FLOOR,
              "OperationType::FLOOR != ANEURALNETWORKS_FLOOR");
static_assert(static_cast<int32_t>(OperationType::FULLY_CONNECTED) ==
                  ANEURALNETWORKS_FULLY_CONNECTED,
              "OperationType::FULLY_CONNECTED != ANEURALNETWORKS_FULLY_CONNECTED");
static_assert(static_cast<int32_t>(OperationType::HASHTABLE_LOOKUP) ==
                  ANEURALNETWORKS_HASHTABLE_LOOKUP,
              "OperationType::HASHTABLE_LOOKUP != ANEURALNETWORKS_HASHTABLE_LOOKUP");
static_assert(static_cast<int32_t>(OperationType::L2_NORMALIZATION) ==
                  ANEURALNETWORKS_L2_NORMALIZATION,
              "OperationType::L2_NORMALIZATION != ANEURALNETWORKS_L2_NORMALIZATION");
static_assert(static_cast<int32_t>(OperationType::L2_POOL_2D) == ANEURALNETWORKS_L2_POOL_2D,
              "OperationType::L2_POOL_2D != ANEURALNETWORKS_L2_POOL_2D");
static_assert(static_cast<int32_t>(OperationType::LOCAL_RESPONSE_NORMALIZATION) ==
                  ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION,
              "OperationType::LOCAL_RESPONSE_NORMALIZATION != "
              "ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION");
static_assert(static_cast<int32_t>(OperationType::LOGISTIC) == ANEURALNETWORKS_LOGISTIC,
              "OperationType::LOGISTIC != ANEURALNETWORKS_LOGISTIC");
static_assert(static_cast<int32_t>(OperationType::LSH_PROJECTION) == ANEURALNETWORKS_LSH_PROJECTION,
              "OperationType::LSH_PROJECTION != ANEURALNETWORKS_LSH_PROJECTION");
static_assert(static_cast<int32_t>(OperationType::LSTM) == ANEURALNETWORKS_LSTM,
              "OperationType::LSTM != ANEURALNETWORKS_LSTM");
static_assert(static_cast<int32_t>(OperationType::MAX_POOL_2D) == ANEURALNETWORKS_MAX_POOL_2D,
              "OperationType::MAX_POOL_2D != ANEURALNETWORKS_MAX_POOL_2D");
static_assert(static_cast<int32_t>(OperationType::MUL) == ANEURALNETWORKS_MUL,
              "OperationType::MUL != ANEURALNETWORKS_MUL");
static_assert(static_cast<int32_t>(OperationType::RELU) == ANEURALNETWORKS_RELU,
              "OperationType::RELU != ANEURALNETWORKS_RELU");
static_assert(static_cast<int32_t>(OperationType::RELU1) == ANEURALNETWORKS_RELU1,
              "OperationType::RELU1 != ANEURALNETWORKS_RELU1");
static_assert(static_cast<int32_t>(OperationType::RELU6) == ANEURALNETWORKS_RELU6,
              "OperationType::RELU6 != ANEURALNETWORKS_RELU6");
static_assert(static_cast<int32_t>(OperationType::RESHAPE) == ANEURALNETWORKS_RESHAPE,
              "OperationType::RESHAPE != ANEURALNETWORKS_RESHAPE");
static_assert(static_cast<int32_t>(OperationType::RESIZE_BILINEAR) ==
                  ANEURALNETWORKS_RESIZE_BILINEAR,
              "OperationType::RESIZE_BILINEAR != ANEURALNETWORKS_RESIZE_BILINEAR");
static_assert(static_cast<int32_t>(OperationType::RNN) == ANEURALNETWORKS_RNN,
              "OperationType::RNN != ANEURALNETWORKS_RNN");
static_assert(static_cast<int32_t>(OperationType::SOFTMAX) == ANEURALNETWORKS_SOFTMAX,
              "OperationType::SOFTMAX != ANEURALNETWORKS_SOFTMAX");
static_assert(static_cast<int32_t>(OperationType::SPACE_TO_DEPTH) == ANEURALNETWORKS_SPACE_TO_DEPTH,
              "OperationType::SPACE_TO_DEPTH != ANEURALNETWORKS_SPACE_TO_DEPTH");
static_assert(static_cast<int32_t>(OperationType::SVDF) == ANEURALNETWORKS_SVDF,
              "OperationType::SVDF != ANEURALNETWORKS_SVDF");
static_assert(static_cast<int32_t>(OperationType::TANH) == ANEURALNETWORKS_TANH,
              "OperationType::TANH != ANEURALNETWORKS_TANH");
