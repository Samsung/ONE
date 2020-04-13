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

#ifndef __MODEL_BUILDER_H__
#define __MODEL_BUILDER_H__

#include "NeuralNetworks.h"

#include "Model.h"

#include "Memory.h"
#include "MemoryTracker.h"

#include <vector>
#include <memory>

class CompilationBuilder;

class ModelBuilder
{
public:
  virtual ~ModelBuilder() = default;

public:
  // Adds an operand to the model.
  int addOperand(const ANeuralNetworksOperandType &type);

public:
  int setOperandValue(uint32_t index, const void *buffer, size_t length);
  int setOperandValueFromMemory(uint32_t index, const Memory *memory, uint32_t offset,
                                size_t length);

public:
  int addOperation(OperationType type, uint32_t inputCount, const uint32_t *inputs,
                   uint32_t outputCount, const uint32_t *outputs);

public:
  int identifyInputsAndOutputs(uint32_t inputCount, const uint32_t *inputs, uint32_t outputCount,
                               const uint32_t *outputs);

public:
  int finish();
  bool isFinished() const { return mCompletedModel; }

public:
  int createCompilation(CompilationBuilder **compilation);

public:
  void publish(Model *model) const;

public:
  uint32_t operandCount() const
  {
    // We don't allow more than uint32_t worth of operands
    return static_cast<uint32_t>(mOperands.size());
  }
  uint32_t operationCount() const
  {
    // We don't allow more than uint32_t worth of operations
    return static_cast<uint32_t>(mOperations.size());
  }

public:
  uint32_t inputCount() const { return static_cast<uint32_t>(mInputIndexes.size()); }
  uint32_t getInputOperandIndex(uint32_t i) const { return mInputIndexes[i]; }
  const Operand &getInputOperand(uint32_t i) const { return mOperands[getInputOperandIndex(i)]; }

public:
  uint32_t outputCount() const { return static_cast<uint32_t>(mOutputIndexes.size()); }
  uint32_t getOutputOperandIndex(uint32_t i) const { return mOutputIndexes[i]; }
  const Operand &getOutputOperand(uint32_t i) const { return mOperands[getOutputOperandIndex(i)]; }

public:
  const Operand &getOperand(uint32_t index) const { return mOperands[index]; }
  const Operation &getOperation(uint32_t index) const { return mOperations[index]; }

public:
  const MemoryTracker &getMemories() const { return mMemories; }
  const std::vector<Operation> &getOperations() const { return mOperations; }

private:
  // Return true if either mCompleteModel or mInvalidModel is true.
  bool badState(const char *name);

  // Sorts the operations to be in the correct order for single threaded
  // node-at-a-time execution.
  void sortIntoRunOrder();

  // Copies the large values to a shared memory, if we have any.
  int copyLargeValuesToMemory();

private:
  // The operations of the graph.
  std::vector<Operation> mOperations;
  // The description of the operands of the graph.
  std::vector<Operand> mOperands;

  // Specifies where to find the list of indexes identifying
  // the inputs and outputs of the model.  The offset is into
  // the mOperandIndexes table.
  std::vector<uint32_t> mInputIndexes;
  std::vector<uint32_t> mOutputIndexes;

  MemoryTracker mMemories;

  // The value of the small operands that are defined at model
  // creation time.
  std::vector<uint8_t> mSmallOperandValues;

  struct LargeValue
  {
    uint32_t operandIndex;
    const void *buffer;
  };
  // Operand index and buffer pointer for all the large operand values of this model.
  std::vector<LargeValue> mLargeOperandValues;
  PrivateMemory mLargeValueMemory;

  // Once the model has been finished, we should not allow further
  // modifications to the model.
  mutable bool mCompletedModel = false;

  // Any invalid manipulation of the model will mark the model invalid.
  // No further modifications are allowed to the model.
  mutable bool mInvalidModel = false;
};

#endif // __MODEL_BUILDER_H__
