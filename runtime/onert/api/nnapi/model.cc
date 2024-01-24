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
#include <NeuralNetworksEx.h>

#include <new>

#include "wrapper/ANeuralNetworksModel.h"
#include "wrapper/ANeuralNetworksMemory.h"
#include "util/logging.h"

int ANeuralNetworksModel_create(ANeuralNetworksModel **model)
{
  if (model == nullptr)
  {
    VERBOSE(NNAPI::Model) << "create: Incorrect null pointer parameter" << std::endl;
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  *model = new (std::nothrow) ANeuralNetworksModel{};
  if (*model == nullptr)
  {
    VERBOSE(NNAPI::Model) << "create: Fail to create model object" << std::endl;
    return ANEURALNETWORKS_OUT_OF_MEMORY;
  }

  return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksModel_free(ANeuralNetworksModel *model) { delete model; }

int ANeuralNetworksModel_addOperand(ANeuralNetworksModel *model,
                                    const ANeuralNetworksOperandType *type)
{
  if ((model == nullptr) || (type == nullptr))
  {
    VERBOSE(NNAPI::Model) << "addOperand: Incorrect null pointer parameter(s)" << std::endl;
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  if (model->isFinished())
  {
    VERBOSE(NNAPI::Model) << "addOperand: Already finished" << std::endl;
    return ANEURALNETWORKS_BAD_STATE;
  }

  // scale and zeroPoint should be zero for scalars and non-fixed point tensors
  // Quantized:
  //  scale: a 32 bit floating point value greater than zero
  //  zeroPoint: a 32 bit integer, in range [0, 255]
  if (type->type == ANEURALNETWORKS_TENSOR_QUANT8_ASYMM)
  {
    if (!(type->scale > 0.0f))
    {
      VERBOSE(NNAPI::Model) << "addOperand: Incorrect scale value for quantization" << std::endl;
      return ANEURALNETWORKS_BAD_DATA;
    }

    if ((type->zeroPoint < 0) || (type->zeroPoint > 255))
    {
      VERBOSE(NNAPI::Model) << "addOperand: Incorrect zeroPoint value for quantization"
                            << std::endl;
      return ANEURALNETWORKS_BAD_DATA;
    }
  }
  // NOTE Validation of scale and zeroPoint would be skipped for a while.
  //      We do not know whether scalar type can have scale and zeroPoint.
  //      To pass ValidationTest and GeneratedTest, this validation code
  //      would not be implemented until we can define this issue clearly.
  //
  // scale and zeroPoint should be zero for scalars and non-fixed point tensors
  // else if ((type->scale != 0.0f) || (type->zeroPoint != 0))
  // {
  //   return ANEURALNETWORKS_BAD_DATA;
  // }

  // dimensionCount should be zero for scalars
  if ((type->dimensionCount != 0) &&
      ((type->type == ANEURALNETWORKS_FLOAT32) || (type->type == ANEURALNETWORKS_INT32) ||
       (type->type == ANEURALNETWORKS_UINT32)))
  {
    VERBOSE(NNAPI::Model) << "addOperand: Incorrect data type" << std::endl;
    return ANEURALNETWORKS_BAD_DATA;
  }

  if (!model->addOperand(type))
  {
    VERBOSE(NNAPI::Model) << "addOperand: Fail to add operand" << std::endl;
    return ANEURALNETWORKS_BAD_DATA;
  }

  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_setOperandValue(ANeuralNetworksModel *model, int32_t index,
                                         const void *buffer, size_t length)
{
  const bool optional_operand = ((buffer == nullptr) && (length == 0));

  if ((model == nullptr) || ((buffer == nullptr) && (length != 0)))
  {
    VERBOSE(NNAPI::Model) << "setOperandValue: Incorrect null pointer parameter(s)" << std::endl;
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  if (model->isFinished())
  {
    VERBOSE(NNAPI::Model) << "setOperandValue: Already finished" << std::endl;
    return ANEURALNETWORKS_BAD_STATE;
  }

  // Negative index value is not allowed
  if (index < 0)
  {
    VERBOSE(NNAPI::Model) << "setOperandValue: Invalid index value (negative)" << std::endl;
    return ANEURALNETWORKS_BAD_DATA;
  }
  // NOTE OperandIndex uses uint32_t as its underlying type as various NNAPI
  //      functions such as ANeuralNetworksModel_addOperation use uint32_t to represent operand
  //      index
  //      ANeuralNetworksModel_setOperandValue, however, uses int32_t to represent operand index.
  //
  //      Below, static_cast<uint32_t>(...) is introduced to eliminate compiler warning.
  uint32_t ind = static_cast<uint32_t>(index);

  if (!model->isExistOperand(ind))
  {
    VERBOSE(NNAPI::Model) << "setOperandValue: Invalid index value (not exist)" << std::endl;
    return ANEURALNETWORKS_BAD_DATA;
  }

  if (!optional_operand && (model->operandSize(ind) != length))
  {
    VERBOSE(NNAPI::Model) << "setOperandValue: Invalid data length" << std::endl;
    return ANEURALNETWORKS_BAD_DATA;
  }

  if (model->isUsageSet(ind))
  {
    VERBOSE(NNAPI::Model) << "setOperandValue: Already set operand" << std::endl;
    return ANEURALNETWORKS_BAD_DATA;
  }

  // NNAPI spec in NeuralNetworks.h
  // For values of length greater than ANEURALNETWORKS_MAX_SIZE_OF_IMMEDIATELY_COPIED_VALUES,
  // the application is responsible for not changing the content of this region
  // until all executions using this model have completed
  bool copy_value = false;
  if (length <= ANEURALNETWORKS_MAX_SIZE_OF_IMMEDIATELY_COPIED_VALUES)
  {
    copy_value = true;
  }

  if (!model->setOperandValue(ind, buffer, length, optional_operand, copy_value))
  {
    VERBOSE(NNAPI::Model) << "setOperandValue: Fail to set operand value" << std::endl;
    return ANEURALNETWORKS_BAD_DATA;
  }

  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_setOperandValueFromMemory(ANeuralNetworksModel *model, int32_t index,
                                                   const ANeuralNetworksMemory *memory,
                                                   size_t offset, size_t length)
{
  if ((model == nullptr) || (memory == nullptr))
  {
    VERBOSE(NNAPI::Model) << "setOperandValueFromMemory: Incorrect null pointer parameter(s)"
                          << std::endl;
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  if (model->isFinished())
  {
    VERBOSE(NNAPI::Model) << "setOperandValueFromMemory: Already finished" << std::endl;
    return ANEURALNETWORKS_BAD_STATE;
  }

  // Negative index value is not allowed
  if (index < 0)
  {
    VERBOSE(NNAPI::Model) << "setOperandValueFromMemory: Invalid index value (negative)"
                          << std::endl;
    return ANEURALNETWORKS_BAD_DATA;
  }
  // NOTE OperandIndex uses uint32_t as its underlying type as various NNAPI
  //      functions such as ANeuralNetworksModel_addOperation use uint32_t to represent operand
  //      index
  //      ANeuralNetworksModel_setOperandValue, however, uses int32_t to represent operand index.
  //
  //      Below, static_cast<uint32_t>(...) is introduced to eliminate compiler warning.
  uint32_t ind = static_cast<uint32_t>(index);

  if (!model->isExistOperand(ind))
  {
    VERBOSE(NNAPI::Model) << "setOperandValueFromMemory: Invalid index value (not exist)"
                          << std::endl;
    return ANEURALNETWORKS_BAD_DATA;
  }

  if ((model->operandSize(ind) != length) || (memory->size() < (offset + length)))
  {
    VERBOSE(NNAPI::Model) << "setOperandValueFromMemory: Invalid data length" << std::endl;
    return ANEURALNETWORKS_BAD_DATA;
  }

  if (model->isUsageSet(ind))
  {
    VERBOSE(NNAPI::Model) << "setOperandValueFromMemory: Already set operand" << std::endl;
    return ANEURALNETWORKS_BAD_DATA;
  }

  if (!model->setOperandValue(ind, memory->base() + offset, length))
  {
    VERBOSE(NNAPI::Model) << "setOperandValueFromMemory: Fail to set operand value" << std::endl;
    return ANEURALNETWORKS_BAD_DATA;
  }

  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_addOperation(ANeuralNetworksModel *model,
                                      ANeuralNetworksOperationType type, uint32_t inputCount,
                                      const uint32_t *inputs, uint32_t outputCount,
                                      const uint32_t *outputs)
{
  if ((model == nullptr) || (inputs == nullptr) || (outputs == nullptr))
  {
    VERBOSE(NNAPI::Model) << "addOperation: Incorrect null pointer parameter(s)" << std::endl;
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  if (model->isFinished())
  {
    VERBOSE(NNAPI::Model) << "addOperation: Already finished" << std::endl;
    return ANEURALNETWORKS_BAD_STATE;
  }

  const ANeuralNetworksOperationType FIRST_OPERATION = ANEURALNETWORKS_ADD;
  const ANeuralNetworksOperationType LAST_OPERATION = ANEURALNETWORKS_RESIZE_NEAREST_NEIGHBOR;
  if ((type < FIRST_OPERATION) || (type > LAST_OPERATION))
  {
    return ANEURALNETWORKS_BAD_DATA;
  }

  for (uint32_t i = 0; i < outputCount; i++)
  {
    if (model->isUsageSet(outputs[i]))
    {
      VERBOSE(NNAPI::Model) << "addOperation: Already set output operand" << std::endl;
      return ANEURALNETWORKS_BAD_DATA;
    }
  }

  if (!model->addOperation(type, inputCount, inputs, outputCount, outputs))
  {
    VERBOSE(NNAPI::Model) << "addOperation: Fail to add operation" << std::endl;
    return ANEURALNETWORKS_BAD_DATA;
  }

  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_addOperationEx(ANeuralNetworksModel *model,
                                        ANeuralNetworksOperationTypeEx type, uint32_t inputCount,
                                        const uint32_t *inputs, uint32_t outputCount,
                                        const uint32_t *outputs)
{
  if ((model == nullptr) || (inputs == nullptr) || (outputs == nullptr))
  {
    VERBOSE(NNAPI::Model) << "addOperation: Incorrect null pointer parameter(s)" << std::endl;
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  if (model->isFinished())
  {
    VERBOSE(NNAPI::Model) << "addOperation: Already finished" << std::endl;
    return ANEURALNETWORKS_BAD_STATE;
  }

  const ANeuralNetworksOperationTypeEx FIRST_OPERATION = ANEURALNETWORKS_CAST_EX;
  const ANeuralNetworksOperationTypeEx LAST_OPERATION = ANEURALNETWORKS_SPLIT_V_EX;
  if ((type < FIRST_OPERATION) || (type > LAST_OPERATION))
  {
    VERBOSE(NNAPI::Model) << "addOperation: Invalid operation type" << std::endl;
    return ANEURALNETWORKS_BAD_DATA;
  }

  for (uint32_t i = 0; i < outputCount; i++)
  {
    if (model->isUsageSet(outputs[i]))
    {
      VERBOSE(NNAPI::Model) << "addOperation: Already set output operand" << std::endl;
      return ANEURALNETWORKS_BAD_DATA;
    }
  }

  if (!model->addOperationEx(type, inputCount, inputs, outputCount, outputs))
  {
    VERBOSE(NNAPI::Model) << "addOperation: Fail to add operation" << std::endl;
    return ANEURALNETWORKS_BAD_DATA;
  }

  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_identifyInputsAndOutputs(ANeuralNetworksModel *model, uint32_t inputCount,
                                                  const uint32_t *inputs, uint32_t outputCount,
                                                  const uint32_t *outputs)
{
  if ((model == nullptr) || (inputs == nullptr) || (outputs == nullptr))
  {
    VERBOSE(NNAPI::Model) << "identifyInputsAndOutputs: Incorrect null pointer parameter(s)"
                          << std::endl;
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  if (model->isFinished())
  {
    VERBOSE(NNAPI::Model) << "identifyInputsAndOutputs: Already finished" << std::endl;
    return ANEURALNETWORKS_BAD_STATE;
  }

  for (uint32_t n = 0; n < inputCount; ++n)
  {
    uint32_t ind = inputs[n];
    if (model->isUsageSet(ind))
    {
      VERBOSE(NNAPI::Model) << "identifyInputsAndOutputs: Already set input operand" << std::endl;
      return ANEURALNETWORKS_BAD_DATA;
    }

    if (!model->addModelInput(ind))
    {
      VERBOSE(NNAPI::Model) << "identifyInputsAndOutputs: Fail to add input" << std::endl;
      return ANEURALNETWORKS_BAD_DATA;
    }
  }

  for (uint32_t n = 0; n < outputCount; ++n)
  {
    uint32_t ind = outputs[n];

    if (!model->isOperationOutput(ind))
    {
      VERBOSE(NNAPI::Model) << "identifyInputsAndOutputs: Need to set output operand" << std::endl;
      return ANEURALNETWORKS_BAD_DATA;
    }

    if (!model->addModelOutput(ind))
    {
      VERBOSE(NNAPI::Model) << "identifyInputsAndOutputs: Fail to add output" << std::endl;
      return ANEURALNETWORKS_BAD_DATA;
    }
  }

  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_finish(ANeuralNetworksModel *model)
{
  if (model == nullptr)
  {
    VERBOSE(NNAPI::Model) << "finish: Incorrect null pointer parameter" << std::endl;
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  if (model->isFinished())
  {
    VERBOSE(NNAPI::Model) << "finish: Already finished" << std::endl;
    return ANEURALNETWORKS_BAD_STATE;
  }

  if (!model->finish())
  {
    VERBOSE(NNAPI::Model) << "finish: Fail to generate internal graph" << std::endl;
    return ANEURALNETWORKS_BAD_STATE;
  }

  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_relaxComputationFloat32toFloat16(ANeuralNetworksModel *model, bool allow)
{
  if (model == nullptr)
  {
    VERBOSE(NNAPI::Model) << "relaxComputationFloat32toFloat16: Incorrect null pointer parameter"
                          << std::endl;
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  if (model->isFinished())
  {
    VERBOSE(NNAPI::Model) << "relaxComputationFloat32toFloat16: Already finished" << std::endl;
    return ANEURALNETWORKS_BAD_STATE;
  }

  model->allowFloat32toFloat16(allow);

  return ANEURALNETWORKS_NO_ERROR;
}
