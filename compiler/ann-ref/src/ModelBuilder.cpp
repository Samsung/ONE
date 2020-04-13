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

#include "ModelBuilder.h"

#include "CompilationBuilder.h"
#include "Validation.h"
#include "Logging.h"
#include "Assert.h"

#include <string.h>
#include <map>

static inline void setFromIntList(std::vector<uint32_t> *vec, uint32_t count, const uint32_t *data)
{
  vec->resize(count);
  for (uint32_t i = 0; i < count; i++)
  {
    (*vec)[i] = data[i];
  }
}

// Returns the number of padding bytes needed to align data of the
// specified length.  It aligns object of length:
// 2, 3 on a 2 byte boundary,
// 4+ on a 4 byte boundary.
// We may want to have different alignments for tensors.
// TODO: This is arbitrary, more a proof of concept.  We need
// to determine what this should be.
uint32_t alignBytesNeeded(uint32_t index, size_t length)
{
  uint32_t pattern;
  if (length < 2)
  {
    pattern = 0; // No alignment necessary
  }
  else if (length < 4)
  {
    pattern = 1; // Align on 2-byte boundary
  }
  else
  {
    pattern = 3; // Align on 4-byte boundary
  }
  uint32_t extra = (~(index - 1)) & pattern;
  return extra;
}


// The maximum number of operands and operations that a model may have.
const uint32_t MAX_NUMBER_OF_OPERANDS = 0xFFFFFFFE;
const uint32_t MAX_NUMBER_OF_OPERATIONS = 0xFFFFFFFE;

bool ModelBuilder::badState(const char *name)
{
  if (mCompletedModel)
  {
    LOG(ERROR) << "ANeuralNetworksModel_" << name << " can't modify after model finished";
    return true;
  }
  if (mInvalidModel)
  {
    LOG(ERROR) << "ANeuralNetworksModel_" << name << " can't modify an invalid model";
    return true;
  }
  return false;
}

int ModelBuilder::addOperand(const ANeuralNetworksOperandType &type)
{
  if (badState("addOperand"))
  {
    return ANEURALNETWORKS_BAD_STATE;
  }

  int n = validateOperandType(type, "ANeuralNetworksModel_addOperand", true);
  if (n != ANEURALNETWORKS_NO_ERROR)
  {
    return n;
  }
  size_t idx = mOperands.size();
  if (idx >= MAX_NUMBER_OF_OPERANDS)
  {
    LOG(ERROR) << "ANeuralNetworksModel_addOperand exceed max operands";
    return ANEURALNETWORKS_BAD_DATA;
  }
  mOperands.resize(idx + 1);
  auto &operand = mOperands[idx];
  operand.type = static_cast<OperandType>(type.type);
  setFromIntList(&operand.dimensions, type.dimensionCount, type.dimensions);
  operand.numberOfConsumers = 0;
  operand.scale = type.scale;
  operand.zeroPoint = type.zeroPoint;
  operand.lifetime = OperandLifeTime::TEMPORARY_VARIABLE;
  operand.location = {.poolIndex = 0, .offset = 0, .length = 0};
  return ANEURALNETWORKS_NO_ERROR;
}

int ModelBuilder::setOperandValue(uint32_t index, const void *buffer, size_t length)
{
  if (badState("setOperandValue"))
  {
    return ANEURALNETWORKS_BAD_STATE;
  }

  VLOG(MODEL) << __func__ << " for operand " << index << " size " << length;
  if (index >= operandCount())
  {
    LOG(ERROR) << "ANeuralNetworksModel_setOperandValue setting operand " << index << " of "
               << operandCount();
    return ANEURALNETWORKS_BAD_DATA;
  }
  Operand &operand = mOperands[index];
  if (buffer == nullptr)
  {
    if (length)
    {
      LOG(ERROR) << "ANeuralNetworksModel_setOperandValue buffer is nullptr but length is "
                    "not 0";
      return ANEURALNETWORKS_BAD_DATA;
    }
    operand.lifetime = OperandLifeTime::NO_VALUE;
    // The location is unused and is set to zeros.
    operand.location = {.poolIndex = 0, .offset = 0, .length = 0};
  }
  else
  {
    if (length > 0xFFFFFFFF)
    {
      LOG(ERROR) << "ANeuralNetworksModel_setOperandValue value length of " << length
                 << " exceeds max size";
      return ANEURALNETWORKS_BAD_DATA;
    }
    uint32_t valueLength = static_cast<uint32_t>(length);
    uint32_t neededLength = sizeOfData(operand.type, operand.dimensions);
    if (neededLength != valueLength)
    {
      LOG(ERROR) << "ANeuralNetworksModel_setOperandValue setting " << valueLength
                 << " bytes when needing " << neededLength;
      return ANEURALNETWORKS_BAD_DATA;
    }
    if (valueLength <= ANEURALNETWORKS_MAX_SIZE_OF_IMMEDIATELY_COPIED_VALUES)
    {
      uint32_t existingSize = static_cast<uint32_t>(mSmallOperandValues.size());
      uint32_t extraBytes = alignBytesNeeded(existingSize, valueLength);
      mSmallOperandValues.resize(existingSize + extraBytes + valueLength);
      operand.lifetime = OperandLifeTime::CONSTANT_COPY;
      operand.location = {
          .poolIndex = 0, .offset = existingSize + extraBytes, .length = neededLength};
      memcpy(&mSmallOperandValues[operand.location.offset], buffer, valueLength);
      VLOG(MODEL) << "Copied small value to offset " << operand.location.offset;
    }
    else
    {
      VLOG(MODEL) << "Saving large value";
      operand.lifetime = OperandLifeTime::CONSTANT_REFERENCE;
      // The values for poolIndex and offset will be set when the model is finished.
      operand.location = {.poolIndex = 0, .offset = 0, .length = valueLength};
      // We keep track of the buffers. We'll allocate the shared memory only
      // once we know the total size, to avoid needless copies.
      mLargeOperandValues.push_back(LargeValue{.operandIndex = index, .buffer = buffer});
    }
  }
  return ANEURALNETWORKS_NO_ERROR;
}

int ModelBuilder::setOperandValueFromMemory(uint32_t index, const Memory *memory, uint32_t offset,
                                            size_t length)
{
  VLOG(MODEL) << __func__ << " for operand " << index << " offset " << offset << " size " << length;
  if (badState("setOperandValueFromMemory"))
  {
    return ANEURALNETWORKS_BAD_STATE;
  }

  if (index >= operandCount())
  {
    LOG(ERROR) << "ANeuralNetworksModel_setOperandValueFromMemory setting operand " << index
               << " of " << operandCount();
    return ANEURALNETWORKS_BAD_DATA;
  }
  Operand &operand = mOperands[index];
  uint32_t neededLength = sizeOfData(operand.type, operand.dimensions);
  if (neededLength != length)
  {
    LOG(ERROR) << "ANeuralNetworksModel_setOperandValueFromMemory setting " << length
               << " bytes when needing " << neededLength;
    return ANEURALNETWORKS_BAD_DATA;
  }
  if (!memory->validateSize(offset, length))
  {
    return ANEURALNETWORKS_BAD_DATA;
  }
  // TODO validate does not exceed length of memory
  operand.lifetime = OperandLifeTime::CONSTANT_REFERENCE;
  operand.location = {.poolIndex = mMemories.add(memory), .offset = offset, .length = neededLength};
  return ANEURALNETWORKS_NO_ERROR;
}

int ModelBuilder::addOperation(OperationType type, uint32_t inputCount, const uint32_t *inputs,
                               uint32_t outputCount, const uint32_t *outputs)
{

  if (badState("addOperation"))
  {
    return ANEURALNETWORKS_BAD_STATE;
  }

  if (!validateOperationType(type))
  {
    LOG(ERROR) << "ANeuralNetworksModel_addOperation invalid operations type "
               << static_cast<uint32_t>(type);
    return ANEURALNETWORKS_BAD_DATA;
  }
  int n = validateOperandList(inputCount, inputs, operandCount(),
                              "ANeuralNetworksModel_addOperation inputs");
  if (n != ANEURALNETWORKS_NO_ERROR)
  {
    return n;
  }
  n = validateOperandList(outputCount, outputs, operandCount(),
                          "ANeuralNetworksModel_addOperation outputs");
  if (n != ANEURALNETWORKS_NO_ERROR)
  {
    return n;
  }

  uint32_t operationIndex = operationCount();
  if (operationIndex >= MAX_NUMBER_OF_OPERATIONS)
  {
    LOG(ERROR) << "ANeuralNetworksModel_addOperation exceed max operations";
    return ANEURALNETWORKS_BAD_DATA;
  }
  mOperations.resize(operationIndex + 1);
  auto &entry = mOperations[operationIndex];
  entry.type = type;

  setFromIntList(&entry.inputs, inputCount, inputs);
  setFromIntList(&entry.outputs, outputCount, outputs);
  for (uint32_t i : entry.inputs)
  {
    mOperands[i].numberOfConsumers++;
    // TODO mOperands[i].consumers.push_back(operationIndex);
  }
  return ANEURALNETWORKS_NO_ERROR;
}

int ModelBuilder::identifyInputsAndOutputs(uint32_t inputCount, const uint32_t *inputs,
                                           uint32_t outputCount, const uint32_t *outputs)
{
  if (badState("identifyInputsAndOutputs"))
  {
    return ANEURALNETWORKS_BAD_STATE;
  }

  int n = validateOperandList(inputCount, inputs, operandCount(),
                              "ANeuralNetworksModel_identifyInputsAndOutputs inputs");
  if (n != ANEURALNETWORKS_NO_ERROR)
  {
    return n;
  }
  n = validateOperandList(outputCount, outputs, operandCount(),
                          "ANeuralNetworksModel_identifyInputsAndOutputs outputs");
  if (n != ANEURALNETWORKS_NO_ERROR)
  {
    return n;
  }

  // Makes a copy of the index list, validates the arguments, and changes
  // the lifetime info of the corresponding operand.
  auto setArguments = [&](std::vector<uint32_t> *indexVector, uint32_t indexCount,
                          const uint32_t *indexList, OperandLifeTime lifetime) -> bool {
    indexVector->resize(indexCount);
    for (uint32_t i = 0; i < indexCount; i++)
    {
      const uint32_t operandIndex = indexList[i];
      if (operandIndex >= mOperands.size())
      {
        LOG(ERROR) << "ANeuralNetworksModel_identifyInputsAndOutputs Can't set input or output "
                      "to be "
                   << operandIndex << " as this exceeds the numbe of operands " << mOperands.size();
        return false;
      }
      (*indexVector)[i] = operandIndex;
      Operand &operand = mOperands[operandIndex];
      if (operand.lifetime != OperandLifeTime::TEMPORARY_VARIABLE)
      {
        LOG(ERROR) << "ANeuralNetworksModel_identifyInputsAndOutputs Can't set operand "
                   << operandIndex
                   << " to be an input or output.  Check that it's not a constant or "
                      "already an input or output";
        return false;
      }
      operand.lifetime = lifetime;
    }
    return true;
  };

  if (!setArguments(&mInputIndexes, inputCount, inputs, OperandLifeTime::MODEL_INPUT) ||
      !setArguments(&mOutputIndexes, outputCount, outputs, OperandLifeTime::MODEL_OUTPUT))
  {
    return ANEURALNETWORKS_BAD_DATA;
  }

  return ANEURALNETWORKS_NO_ERROR;
}

int ModelBuilder::createCompilation(CompilationBuilder **compilation)
{
  if (!mCompletedModel || mInvalidModel)
  {
    LOG(ERROR) << "ANeuralNetworksCompilation_create passed an unfinished model";
    *compilation = nullptr;
    return ANEURALNETWORKS_BAD_STATE;
  }
  *compilation = new CompilationBuilder(this);
  return (*compilation ? ANEURALNETWORKS_NO_ERROR : ANEURALNETWORKS_OUT_OF_MEMORY);
}

int ModelBuilder::finish()
{
  if (mCompletedModel)
  {
    LOG(ERROR) << "ANeuralNetworksModel_finish called more than once";
    return ANEURALNETWORKS_BAD_STATE;
  }
  if (mInvalidModel)
  {
    LOG(ERROR) << "ANeuralNetworksModel_finish called on an invalid model";
    return ANEURALNETWORKS_BAD_STATE;
  }

  int n = copyLargeValuesToMemory();
  if (n != ANEURALNETWORKS_NO_ERROR)
  {
    return n;
  }

  Model modelForValidation;
  publish(&modelForValidation);
  if (!validateModel(modelForValidation))
  {
    LOG(ERROR) << "ANeuralNetworksModel_finish called on invalid model";
    mInvalidModel = true;
    return ANEURALNETWORKS_BAD_DATA;
  }

  // We sort the operations so that they will be in the appropriate
  // order for a single-threaded, op at a time execution.
  // TODO: we don't need this if we always run the partitioner.
  sortIntoRunOrder();
  mCompletedModel = true;
  return ANEURALNETWORKS_NO_ERROR;
}

void ModelBuilder::sortIntoRunOrder()
{
  // Tracks the operations that can be executed.
  std::vector<uint32_t> opsReadyToRun;
  std::vector<Operation> runOrder;

  // Tracks how many inputs are needed for each operation to be ready to run.
  std::multimap<uint32_t, uint32_t> operandToOperations;
  std::vector<uint32_t> unknownInputCount(operationCount());
  for (uint32_t operationIndex = 0; operationIndex < operationCount(); operationIndex++)
  {
    uint32_t &count = unknownInputCount[operationIndex];
    count = 0;
    for (uint32_t operandIndex : mOperations[operationIndex].inputs)
    {
      auto lifetime = mOperands[operandIndex].lifetime;
      if (lifetime == OperandLifeTime::TEMPORARY_VARIABLE ||
          lifetime == OperandLifeTime::MODEL_OUTPUT)
      {
        count++;
        operandToOperations.insert(std::pair<uint32_t, uint32_t>(operandIndex, operationIndex));
      }
    }
    if (count == 0)
    {
      opsReadyToRun.push_back(operationIndex);
    }
  }

  while (opsReadyToRun.size() > 0)
  {
    // Execute the next op
    int opIndex = opsReadyToRun.back();
    opsReadyToRun.pop_back();
    const Operation &operation = mOperations[opIndex];

    runOrder.push_back(mOperations[opIndex]);

    // Mark all its outputs as known.
    for (uint32_t operandIndex : operation.outputs)
    {
      auto range = operandToOperations.equal_range(operandIndex);
      for (auto i = range.first; i != range.second; i++)
      {
        uint32_t &count = unknownInputCount[i->second];
        if (--count == 0)
        {
          opsReadyToRun.push_back(i->second);
        }
      }
    }
  }
  mOperations = runOrder;
}

void ModelBuilder::publish(Model *model) const
{
  model->operands = mOperands;
  model->operations = mOperations;
  model->inputIndexes = mInputIndexes;
  model->outputIndexes = mOutputIndexes;
  model->operandValues = mSmallOperandValues;

  uint32_t count = mMemories.size();
  model->pools.resize(count);
  for (uint32_t i = 0; i < count; i++)
  {
    uint8_t *buffer;
    mMemories[i]->getPointer(&buffer);
    model->pools[i] = buffer;
  }
}

int ModelBuilder::copyLargeValuesToMemory()
{
  if (!mLargeOperandValues.empty())
  {
    // Calculate the size of the shared memory needed for all the large values.
    // Also sets the offset for each value within the memory.
    size_t poolSize = 0;
    for (LargeValue &l : mLargeOperandValues)
    {
      Operand &operand = mOperands[l.operandIndex];
      ASSERT(operand.lifetime == OperandLifeTime::CONSTANT_REFERENCE);
      poolSize += alignBytesNeeded(poolSize, operand.location.length);
      operand.location.offset = poolSize;
      poolSize += operand.location.length;
    }

    // Allocated the shared memory.
    int n = mLargeValueMemory.create(poolSize);
    if (n != ANEURALNETWORKS_NO_ERROR)
    {
      return n;
    }
    uint8_t *memoryPointer = nullptr;
    n = mLargeValueMemory.getPointer(&memoryPointer);
    if (n != ANEURALNETWORKS_NO_ERROR)
    {
      return n;
    }
    uint32_t poolIndex = mMemories.add(&mLargeValueMemory);
    VLOG(MODEL) << "Allocated large value pool of size " << poolSize << " at index " << poolIndex;

    // Copy the values to this memory.
    for (LargeValue &l : mLargeOperandValues)
    {
      Operand &operand = mOperands[l.operandIndex];
      operand.location.poolIndex = poolIndex;
      memcpy(memoryPointer + operand.location.offset, l.buffer, operand.location.length);
    }
  }
  return ANEURALNETWORKS_NO_ERROR;
}
