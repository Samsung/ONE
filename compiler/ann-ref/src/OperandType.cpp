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
#include "Macro.h"

const char *kTypeNames[] = {
    "FLOAT32", "INT32", "UINT32", "TENSOR_FLOAT32", "TENSOR_INT32", "TENSOR_QUANT8_ASYMM",
};

static_assert(COUNT(kTypeNames) == kNumberOfDataTypes, "kTypeNames is incorrect");

const uint32_t kSizeOfDataType[]{
    4, // ANEURALNETWORKS_FLOAT32
    4, // ANEURALNETWORKS_INT32
    4, // ANEURALNETWORKS_UINT32
    4, // ANEURALNETWORKS_TENSOR_FLOAT32
    4, // ANEURALNETWORKS_TENSOR_INT32
    1  // ANEURALNETWORKS_TENSOR_SYMMETRICAL_QUANT8
};

static_assert(COUNT(kSizeOfDataType) == kNumberOfDataTypes, "kSizeOfDataType is incorrect");

const char *getOperandTypeName(OperandType type)
{
  uint32_t n = static_cast<uint32_t>(type);
  return kTypeNames[n];
}

uint32_t sizeOfData(OperandType type, const std::vector<uint32_t> &dimensions)
{
  int n = static_cast<int>(type);

  uint32_t size = kSizeOfDataType[n];

  for (auto d : dimensions)
  {
    size *= d;
  }
  return size;
}
