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

#include "ModelArgumentInfo.h"
#include "NeuralNetworks.h" // For ANEURALNETWORKS_XXX
#include "Logging.h"
#include "Assert.h"

// TODO-NNRT: Consider removing ModelArgumentInfo completely if it's not necessary
int ModelArgumentInfo::setFromPointer(const Operand &operand,
                                      const ANeuralNetworksOperandType *type, void *data,
                                      uint32_t length)
{
  if ((data == nullptr) != (length == 0))
  {
    LOG(ERROR) << "Data pointer must be nullptr if and only if length is zero (data = " << data
               << ", length = " << length << ")";
    return ANEURALNETWORKS_BAD_DATA;
  }
  if (data == nullptr)
  {
    state = ModelArgumentInfo::HAS_NO_VALUE;
  }
  else
  {
    int n = updateDimensionInfo(operand, type);
    if (n != ANEURALNETWORKS_NO_ERROR)
    {
      return n;
    }
    uint32_t neededLength = sizeOfData(operand.type, dimensions);
    if (neededLength != length)
    {
      LOG(ERROR) << "Setting argument with invalid length: " << length
                 << ", expected length: " << neededLength;
      return ANEURALNETWORKS_BAD_DATA;
    }
    state = ModelArgumentInfo::POINTER;
  }
  buffer = data;
  locationAndLength = {.poolIndex = 0, .offset = 0, .length = length};
  return ANEURALNETWORKS_NO_ERROR;
}

int ModelArgumentInfo::setFromMemory(const Operand &operand, const ANeuralNetworksOperandType *type,
                                     uint32_t poolIndex, uint32_t offset, uint32_t length)
{
  int n = updateDimensionInfo(operand, type);
  if (n != ANEURALNETWORKS_NO_ERROR)
  {
    return n;
  }
  uint32_t neededLength = sizeOfData(operand.type, dimensions);
  if (neededLength != length)
  {
    LOG(ERROR) << "Setting argument with invalid length: " << length
               << ", expected length: " << neededLength;
    return ANEURALNETWORKS_BAD_DATA;
  }

  state = ModelArgumentInfo::MEMORY;
  locationAndLength = {.poolIndex = poolIndex, .offset = offset, .length = length};
  buffer = nullptr;
  return ANEURALNETWORKS_NO_ERROR;
}

int ModelArgumentInfo::updateDimensionInfo(const Operand &operand,
                                           const ANeuralNetworksOperandType *newType)
{
  ASSERT(dimensions.empty());
  if (newType == nullptr)
  {
    for (auto i : operand.dimensions)
    {
      if (i == 0)
      {
        LOG(ERROR) << "Setting input/output with unspecified dimensions";
        return ANEURALNETWORKS_BAD_DATA;
      }
    }
    dimensions = operand.dimensions;
  }
  else
  {
    uint32_t count = newType->dimensionCount;
    if (static_cast<OperandType>(newType->type) != operand.type ||
        count != operand.dimensions.size())
    {
      LOG(ERROR) << "Setting input/output with incompatible types";
      return ANEURALNETWORKS_BAD_DATA;
    }
    dimensions = std::vector<uint32_t>(count);
    for (uint32_t i = 0; i < count; i++)
    {
      if (operand.dimensions[i] != 0 && operand.dimensions[i] != newType->dimensions[i])
      {
        LOG(ERROR) << "Overriding a fully specified dimension is disallowed";
        return ANEURALNETWORKS_BAD_DATA;
      }
      else
      {
        dimensions[i] = newType->dimensions[i];
      }
    }
  }
  return ANEURALNETWORKS_NO_ERROR;
}
