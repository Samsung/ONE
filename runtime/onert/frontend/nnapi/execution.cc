/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <NeuralNetworks.h>

#include <new>

#include "wrapper/ANeuralNetworksCompilation.h"
#include "wrapper/ANeuralNetworksExecution.h"
#include "wrapper/ANeuralNetworksMemory.h"
#include "wrapper/ANeuralNetworksEvent.h"
#include "wrapper/NNAPIConvert.h"
#include "util/logging.h"

//
// NNAPI Implementation
//
int ANeuralNetworksExecution_create(ANeuralNetworksCompilation *compilation,
                                    ANeuralNetworksExecution **execution)
{
  if ((compilation == nullptr) || (execution == nullptr))
  {
    VERBOSE(NNAPI::Execution) << "create: Incorrect null pointer parameter(s)" << std::endl;
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  std::shared_ptr<onert::exec::IExecutors> executors;

  compilation->publish(executors);

  if (executors == nullptr)
  {
    VERBOSE(NNAPI::Execution) << "create: Never compiled yet" << std::endl;
    return ANEURALNETWORKS_BAD_STATE;
  }

  *execution = new (std::nothrow) ANeuralNetworksExecution{executors};
  if (*execution == nullptr)
  {
    VERBOSE(NNAPI::Execution) << "create: Fail to create execution object" << std::endl;
    return ANEURALNETWORKS_OUT_OF_MEMORY;
  }

  return ANEURALNETWORKS_NO_ERROR;
}

// NOTE Handle optional input
//  Unspecified shape on model build
//    Optional and omitted input on execution: skip input setting (workaround for LSTM)
//    Optional but not omitted input on execution: cannot handle
//    Normal input on execution: cannot handle
//  Fully specified shape on model build
//    Optional input on execution: cannot handle
//    Normal input: handle normally
int ANeuralNetworksExecution_setInput(ANeuralNetworksExecution *execution, int32_t index,
                                      const ANeuralNetworksOperandType *type, const void *buffer,
                                      size_t length)
{
  // Don't check type
  // Comment about ANeuralNetworksOperandType in NeuralNetworks.h:
  //  If the input or output is optional and omitted then it need not have a fully specified tensor
  //  operand type
  if ((execution == nullptr) || ((buffer == nullptr) && (length != 0)))
  {
    VERBOSE(NNAPI::Execution) << "setInput: Incorrect null pointer parameter(s)" << std::endl;
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  if ((buffer != nullptr) && (length == 0))
  {
    VERBOSE(NNAPI::Execution) << "setInput: Zero length input" << std::endl;
    return ANEURALNETWORKS_BAD_DATA;
  }

  const auto operand_index = execution->getInputOperandIndex(index);
  if (!operand_index.valid())
  {
    VERBOSE(NNAPI::Execution) << "setInput: Invalid input index" << std::endl;
    return ANEURALNETWORKS_BAD_DATA;
  }

  // Omitted optional input
  // LSTM operation's some inputs can be optional input
  // Transpose operation's permutation input can be optional input
  if ((buffer == nullptr) && (length == 0))
  {
    uint32_t dims[1] = {0};
    ANeuralNetworksOperandType compared_shape;
    compared_shape.dimensionCount = 1;
    compared_shape.dimensions = dims;
    if (execution->hasUnspecifiedDims(operand_index))
    {
      return ANEURALNETWORKS_NO_ERROR;
    }
    else if (type == nullptr && execution->IsOptionalInput(operand_index))
    {
      if (!execution->setOptionalInput(index, type, buffer, length))
      {
        VERBOSE(NNAPI::Execution) << "setInput: Fail to set optional input" << std::endl;
        return ANEURALNETWORKS_BAD_DATA;
      }
      return ANEURALNETWORKS_NO_ERROR;
    }
    // TODO Changes the condition to check zero sized
    else if (execution->compareShape(&compared_shape, operand_index))
    {
      if (!execution->setInput(index, type, buffer, length))
      {
        VERBOSE(NNAPI::Execution) << "setInput: Fail to set input" << std::endl;
        return ANEURALNETWORKS_BAD_DATA;
      }
      return ANEURALNETWORKS_NO_ERROR;
    }
    else
    {
      VERBOSE(NNAPI::Execution) << "setInput: Cannot handle fully-specified shape on model build "
                                   "but omitted input on execution"
                                << std::endl;
      return ANEURALNETWORKS_BAD_DATA;
    }
  }

  if (type != nullptr)
  {
    if (!execution->compareDataType(type, operand_index))
    {
      VERBOSE(NNAPI::Execution) << "setInput: Data type mismatch" << std::endl;
      return ANEURALNETWORKS_BAD_DATA;
    }

    if (!execution->compareShape(type, operand_index))
    {
      VERBOSE(NNAPI::Execution) << "setInput: Shape mismatch" << std::endl;
      return ANEURALNETWORKS_BAD_DATA;
    }

    if (NNAPIConvert::calculateSizeFromType(type) != length)
    {
      VERBOSE(NNAPI::Execution) << "setInput: Invalid length" << std::endl;
      return ANEURALNETWORKS_BAD_DATA;
    }
  }
  else
  {
    if (execution->hasUnspecifiedDims(operand_index))
    {
      VERBOSE(NNAPI::Execution) << "setInput: Unspecified dimension value" << std::endl;
      return ANEURALNETWORKS_BAD_DATA;
    }

    if (execution->getOperandSize(operand_index) != length)
    {
      VERBOSE(NNAPI::Execution) << "setInput: Invalid length" << std::endl;
      return ANEURALNETWORKS_BAD_DATA;
    }
  }

  if (!execution->setInput(index, type, buffer, length))
  {
    VERBOSE(NNAPI::Execution) << "setInput: Fail to set input" << std::endl;
    return ANEURALNETWORKS_BAD_DATA;
  }

  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksExecution_setOutput(ANeuralNetworksExecution *execution, int32_t index,
                                       const ANeuralNetworksOperandType *type, void *buffer,
                                       size_t length)
{
  // Don't check type
  // Comment about ANeuralNetworksOperandType in NeuralNetworks.h:
  //  If the input or output is optional and omitted then it need not have a fully specified tensor
  //  operand type
  if ((execution == nullptr) || ((buffer == nullptr) && (length != 0)))
  {
    VERBOSE(NNAPI::Execution) << "setOutput: Incorrect null pointer parameter(s)" << std::endl;
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  if ((buffer != nullptr) && (length == 0))
  {
    VERBOSE(NNAPI::Execution) << "setOutput: Zero length output" << std::endl;
    return ANEURALNETWORKS_BAD_DATA;
  }

  // Handle optional output
  if (buffer == nullptr)
  {
    return ANEURALNETWORKS_NO_ERROR;
  }

  const auto operand_index = execution->getOutputOperandIndex(index);
  if (!operand_index.valid())
  {
    VERBOSE(NNAPI::Execution) << "setOutput: Invalid output index" << std::endl;
    return ANEURALNETWORKS_BAD_DATA;
  }

  if (type != nullptr)
  {
    if (!execution->compareDataType(type, operand_index))
    {
      VERBOSE(NNAPI::Execution) << "setOutput: Data type mismatch" << std::endl;
      return ANEURALNETWORKS_BAD_DATA;
    }

    if (!execution->compareShape(type, operand_index))
    {
      VERBOSE(NNAPI::Execution) << "setOutput: Shape mismatch" << std::endl;
      return ANEURALNETWORKS_BAD_DATA;
    }

    if (NNAPIConvert::calculateSizeFromType(type) != length)
    {
      VERBOSE(NNAPI::Execution) << "setOutput: Invalid length" << std::endl;
      return ANEURALNETWORKS_BAD_DATA;
    }
  }
  else
  {
    if (execution->hasUnspecifiedDims(operand_index))
    {
      VERBOSE(NNAPI::Execution) << "setOutput: Unspecified dimension value" << std::endl;
      return ANEURALNETWORKS_BAD_DATA;
    }

    if (execution->getOperandSize(operand_index) != length)
    {
      VERBOSE(NNAPI::Execution) << "setOutput: Invalid length" << std::endl;
      return ANEURALNETWORKS_BAD_DATA;
    }
  }

  if (!execution->setOutput(index, type, buffer, length))
  {
    VERBOSE(NNAPI::Execution) << "setOutput: Fail to set output" << std::endl;
    return ANEURALNETWORKS_BAD_DATA;
  }

  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksExecution_startCompute(ANeuralNetworksExecution *execution,
                                          ANeuralNetworksEvent **event)
{
  if ((execution == nullptr) || (event == nullptr))
  {
    VERBOSE(NNAPI::Execution) << "startCompute: Incorrect null pointer parameter(s)" << std::endl;
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  // TODO: Handle event
  auto instance = execution->instance();
  *event = new (std::nothrow) ANeuralNetworksEvent{instance};
  if (*event == nullptr)
  {
    VERBOSE(NNAPI::Execution) << "startCompute: Fail to create event" << std::endl;
    return ANEURALNETWORKS_OUT_OF_MEMORY;
  }

  if (!execution->startExecute())
  {
    VERBOSE(NNAPI::Execution) << "startCompute: Fail to start execution" << std::endl;
    return ANEURALNETWORKS_BAD_STATE;
  }

  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksExecution_compute(ANeuralNetworksExecution *execution)
{
  if (execution == nullptr)
  {
    VERBOSE(NNAPI::Execution) << "Compute: Incorrect null pointer parameter" << std::endl;
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  if (!execution->execute())
  {
    VERBOSE(NNAPI::Execution) << "Compute: Fail to execution" << std::endl;
    return ANEURALNETWORKS_BAD_STATE;
  }

  return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksExecution_free(ANeuralNetworksExecution *execution) { delete execution; }

int ANeuralNetworksExecution_setInputFromMemory(ANeuralNetworksExecution *execution, int32_t index,
                                                const ANeuralNetworksOperandType *type,
                                                const ANeuralNetworksMemory *memory, size_t offset,
                                                size_t length)
{
  if ((execution == nullptr) || (memory == nullptr))
  {
    VERBOSE(NNAPI::Execution) << "setInputFromMemory: Incorrect null pointer parameter(s)"
                              << std::endl;
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  if (length == 0)
  {
    VERBOSE(NNAPI::Execution) << "setInputFromMemory: Zero length input" << std::endl;
    return ANEURALNETWORKS_BAD_DATA;
  }

  const auto operand_index = execution->getInputOperandIndex(index);
  if (!operand_index.valid())
  {
    VERBOSE(NNAPI::Execution) << "setInputFromMemory: Invalid input index" << std::endl;
    return ANEURALNETWORKS_BAD_DATA;
  }

  if (type != nullptr)
  {
    if (!execution->compareDataType(type, operand_index))
    {
      VERBOSE(NNAPI::Execution) << "setInputFromMemory: Data type mismatch" << std::endl;
      return ANEURALNETWORKS_BAD_DATA;
    }

    if (!execution->compareShape(type, operand_index))
    {
      VERBOSE(NNAPI::Execution) << "setInputFromMemory: Shape mismatch" << std::endl;
      return ANEURALNETWORKS_BAD_DATA;
    }

    if (NNAPIConvert::calculateSizeFromType(type) != length)
    {
      VERBOSE(NNAPI::Execution) << "setInputFromMemory: Invalid length" << std::endl;
      return ANEURALNETWORKS_BAD_DATA;
    }
  }
  else
  {
    if (execution->hasUnspecifiedDims(operand_index))
    {
      VERBOSE(NNAPI::Execution) << "setInputFromMemory: Unspecified dimension value" << std::endl;
      return ANEURALNETWORKS_BAD_DATA;
    }

    if (execution->getOperandSize(operand_index) != length)
    {
      VERBOSE(NNAPI::Execution) << "setInputFromMemory: Invalid length" << std::endl;
      return ANEURALNETWORKS_BAD_DATA;
    }
  }

  if (!memory->vaildAccess(offset, length))
  {
    VERBOSE(NNAPI::Execution) << "setInputFromMemory: Invalid memory access" << std::endl;
    return ANEURALNETWORKS_BAD_DATA;
  }

  if (!execution->setInput(index, type, reinterpret_cast<const void *>(memory->base() + offset),
                           length))
  {
    VERBOSE(NNAPI::Execution) << "setInputFromMemory: Fail to set input" << std::endl;
    return ANEURALNETWORKS_BAD_DATA;
  }

  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksExecution_setOutputFromMemory(ANeuralNetworksExecution *execution, int32_t index,
                                                 const ANeuralNetworksOperandType *type,
                                                 const ANeuralNetworksMemory *memory, size_t offset,
                                                 size_t length)
{
  if ((execution == nullptr) || (memory == nullptr))
  {
    VERBOSE(NNAPI::Execution) << "setOutputFromMemory: Incorrect null pointer parameter(s)"
                              << std::endl;
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  if (length == 0)
  {
    VERBOSE(NNAPI::Execution) << "setOutputFromMemory: Zero length input" << std::endl;
    return ANEURALNETWORKS_BAD_DATA;
  }

  const auto operand_index = execution->getOutputOperandIndex(index);
  if (!operand_index.valid())
  {
    VERBOSE(NNAPI::Execution) << "setOutputFromMemory: Invalid output index" << std::endl;
    return ANEURALNETWORKS_BAD_DATA;
  }

  if (type != nullptr)
  {
    if (!execution->compareDataType(type, operand_index))
    {
      VERBOSE(NNAPI::Execution) << "setOutputFromMemory: Data type mismatch" << std::endl;
      return ANEURALNETWORKS_BAD_DATA;
    }

    if (!execution->compareShape(type, operand_index))
    {
      VERBOSE(NNAPI::Execution) << "setOutputFromMemory: Shape mismatch" << std::endl;
      return ANEURALNETWORKS_BAD_DATA;
    }

    if (NNAPIConvert::calculateSizeFromType(type) != length)
    {
      VERBOSE(NNAPI::Execution) << "setOutputFromMemory: Invalid length" << std::endl;
      return ANEURALNETWORKS_BAD_DATA;
    }
  }
  else
  {
    if (execution->hasUnspecifiedDims(operand_index))
    {
      VERBOSE(NNAPI::Execution) << "setOutputFromMemory: Unspecified dimension value" << std::endl;
      return ANEURALNETWORKS_BAD_DATA;
    }

    if (execution->getOperandSize(operand_index) != length)
    {
      VERBOSE(NNAPI::Execution) << "setOutputFromMemory: Invalid length" << std::endl;
      return ANEURALNETWORKS_BAD_DATA;
    }
  }

  if (!memory->vaildAccess(offset, length))
  {
    VERBOSE(NNAPI::Execution) << "setOutputFromMemory: Invalid memory access" << std::endl;
    return ANEURALNETWORKS_BAD_DATA;
  }

  if (!execution->setOutput(index, type, reinterpret_cast<void *>(memory->base() + offset), length))
  {
    VERBOSE(NNAPI::Execution) << "setOutputFromMemory: Fail to set input" << std::endl;
    return ANEURALNETWORKS_BAD_DATA;
  }

  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksExecution_getOutputOperandRank(ANeuralNetworksExecution *execution,
                                                  int32_t index, uint32_t *rank)
{
  if ((execution == nullptr) || (rank == nullptr))
  {
    VERBOSE(NNAPI::Execution) << "getOutputOperandRank: Incorrect null pointer parameter(s)"
                              << std::endl;
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  const auto operand_index = execution->getOutputOperandIndex(index);
  if (!operand_index.valid())
  {
    VERBOSE(NNAPI::Execution) << "getOutputOperandRank: Invalid output index" << std::endl;
    return ANEURALNETWORKS_BAD_DATA;
  }

  if (!execution->getOutputOperandRank(index, rank))
  {
    VERBOSE(NNAPI::Execution) << "getOutputOperandRank: Fail to get rank" << std::endl;
    return ANEURALNETWORKS_BAD_STATE;
  }

  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksExecution_getOutputOperandDimensions(ANeuralNetworksExecution *execution,
                                                        int32_t index, uint32_t *dimensions)
{
  if ((execution == nullptr) || (dimensions == nullptr))
  {
    VERBOSE(NNAPI::Execution) << "getOutputOperandDimensions: Incorrect null pointer parameter(s)"
                              << std::endl;
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  const auto operand_index = execution->getOutputOperandIndex(index);
  if (!operand_index.valid())
  {
    VERBOSE(NNAPI::Execution) << "getOutputOperandDimensions: Invalid output index" << std::endl;
    return ANEURALNETWORKS_BAD_DATA;
  }

  if (!execution->getOutputOperandDimensions(index, dimensions))
  {
    VERBOSE(NNAPI::Execution) << "getOutputOperandDimensions: Fail to get rank" << std::endl;
    return ANEURALNETWORKS_BAD_STATE;
  }

  return ANEURALNETWORKS_NO_ERROR;
}
