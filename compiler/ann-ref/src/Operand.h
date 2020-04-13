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

#ifndef __OPERAND_H__
#define __OPERAND_H__

#include "OperandType.h"

#include <cstdint>
#include <vector>

enum class OperandLifeTime : int32_t {
  TEMPORARY_VARIABLE = 0,
  MODEL_INPUT = 1,
  MODEL_OUTPUT = 2,
  CONSTANT_COPY = 3,
  CONSTANT_REFERENCE = 4,
  NO_VALUE = 5,
};

struct DataLocation final {
  uint32_t poolIndex;
  uint32_t offset;
  uint32_t length;
};

struct Operand final {
  OperandType type;
  float scale;
  int32_t zeroPoint;

  std::vector<uint32_t> dimensions;

  DataLocation location;

  uint32_t numberOfConsumers;
  OperandLifeTime lifetime;
};

// Returns the amount of space needed to store a value of the dimensions and
// type of this operand.
inline uint32_t sizeOfData(const Operand &operand)
{
  return sizeOfData(operand.type, operand.dimensions);
}

#endif // __OPERAND_H__
