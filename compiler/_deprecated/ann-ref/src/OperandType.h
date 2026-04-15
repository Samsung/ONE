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

#ifndef __OPERAND_TYPES_H__
#define __OPERAND_TYPES_H__

#include <cstdint>
#include <vector>

enum class OperandType : int32_t {
  FLOAT32 = 0,
  INT32 = 1,
  UINT32 = 2,
  TENSOR_FLOAT32 = 3,
  TENSOR_INT32 = 4,
  TENSOR_QUANT8_ASYMM = 5,
};

// The number of data types (OperandCode) defined in NeuralNetworks.h.
const int kNumberOfDataTypes = 6;

// Returns the name of the operand type in ASCII.
const char *getOperandTypeName(OperandType type);

// Returns the amount of space needed to store a value of the specified
// dimensions and type.
uint32_t sizeOfData(OperandType type, const std::vector<uint32_t> &dimensions);

#endif // __OPERAND_TYPES_H__
