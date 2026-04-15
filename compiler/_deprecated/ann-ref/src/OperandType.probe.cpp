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

#include "OperandType.h"
#include "NeuralNetworks.h"

static_assert(static_cast<int32_t>(OperandType::FLOAT32) == ANEURALNETWORKS_FLOAT32,
              "FLOAT32 != ANEURALNETWORKS_FLOAT32");
static_assert(static_cast<int32_t>(OperandType::INT32) == ANEURALNETWORKS_INT32,
              "INT32 != ANEURALNETWORKS_INT32");
static_assert(static_cast<int32_t>(OperandType::UINT32) == ANEURALNETWORKS_UINT32,
              "UINT32 != ANEURALNETWORKS_UINT32");

static_assert(static_cast<int32_t>(OperandType::TENSOR_FLOAT32) == ANEURALNETWORKS_TENSOR_FLOAT32,
              "TENSOR_FLOAT32 != ANEURALNETWORKS_TENSOR_FLOAT32");
static_assert(static_cast<int32_t>(OperandType::TENSOR_QUANT8_ASYMM) ==
                  ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
              "TENSOR_QUANT8_ASYMM != ANEURALNETWORKS_TENSOR_QUANT8_ASYMM");
