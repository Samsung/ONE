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

#ifndef __MODEL_ARGUMENT_INFO_H__
#define __MODEL_ARGUMENT_INFO_H__

#include "NeuralNetworks.h"

#include "Operand.h"

#include <vector>

struct ModelArgumentInfo
{
  // Whether the argument was specified as being in a Memory, as a pointer,
  // has no value, or has not been specified.
  // If POINTER then:
  //   locationAndLength.length is valid.
  //   dimensions is valid.
  //   buffer is valid
  // If MEMORY then:
  //   locationAndLength.location.{poolIndex, offset, length} is valid.
  //   dimensions is valid.
  enum
  {
    POINTER,
    MEMORY,
    HAS_NO_VALUE,
    UNSPECIFIED
  } state = UNSPECIFIED;

  DataLocation locationAndLength;

  std::vector<uint32_t> dimensions;
  void *buffer;

  int setFromPointer(const Operand &operand, const ANeuralNetworksOperandType *type, void *buffer,
                     uint32_t length);
  int setFromMemory(const Operand &operand, const ANeuralNetworksOperandType *type,
                    uint32_t poolIndex, uint32_t offset, uint32_t length);
  int updateDimensionInfo(const Operand &operand, const ANeuralNetworksOperandType *newType);
};

#endif // __MODEL_ARGUMENT_INFO_H__
