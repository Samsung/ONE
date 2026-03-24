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

#include "Validation.h"
#include "Macro.h"
#include "Assert.h"

static inline bool validCode(uint32_t codeCount, uint32_t code)
{
  return (code < codeCount);
}

int validateOperationType(const OperationType &type)
{
  return validCode(kNumberOfOperationTypes, static_cast<uint32_t>(type));
}

// Validates the type. The used dimensions can be underspecified.
int validateOperandType(const ANeuralNetworksOperandType &type, const char *tag, bool allowPartial)
{
  if (!allowPartial)
  {
    for (uint32_t i = 0; i < type.dimensionCount; i++)
    {
      if (type.dimensions[i] == 0)
      {
        LOG(ERROR) << tag << " OperandType invalid dimensions[" << i
                   << "] = " << type.dimensions[i];
        return ANEURALNETWORKS_BAD_DATA;
      }
    }
  }
  if (!validCode(kNumberOfDataTypes, type.type))
  {
    LOG(ERROR) << tag << " OperandType invalid type " << type.type;
    return ANEURALNETWORKS_BAD_DATA;
  }
  if (type.type == ANEURALNETWORKS_TENSOR_QUANT8_ASYMM)
  {
    if (type.zeroPoint < 0 || type.zeroPoint > 255)
    {
      LOG(ERROR) << tag << " OperandType invalid zeroPoint " << type.zeroPoint;
      return ANEURALNETWORKS_BAD_DATA;
    }
    if (type.scale < 0.f)
    {
      LOG(ERROR) << tag << " OperandType invalid scale " << type.scale;
      return ANEURALNETWORKS_BAD_DATA;
    }
  }

  // TODO-NNRT : add 'type.type == ANEURALNETWORKS_OEM_SCALAR' later.
  //             OEM operaters are not supported now.
  if (type.type == ANEURALNETWORKS_FLOAT32 || type.type == ANEURALNETWORKS_INT32 ||
      type.type == ANEURALNETWORKS_UINT32)
  {
    if (type.dimensionCount != 0 || type.dimensions != nullptr)
    {
      LOG(ERROR) << tag << " Invalid dimensions for scalar type";
      return ANEURALNETWORKS_BAD_DATA;
    }
  }

  return ANEURALNETWORKS_NO_ERROR;
}

int validateOperandList(uint32_t count, const uint32_t *list, uint32_t operandCount,
                        const char *tag)
{
  for (uint32_t i = 0; i < count; i++)
  {
    if (list[i] >= operandCount)
    {
      LOG(ERROR) << tag << " invalid operand index at " << i << " = " << list[i]
                 << ", operandCount " << operandCount;
      return ANEURALNETWORKS_BAD_DATA;
    }
  }
  return ANEURALNETWORKS_NO_ERROR;
}

static bool validOperandIndexes(const std::vector<uint32_t> indexes, size_t operandCount)
{
  for (uint32_t i : indexes)
  {
    if (i >= operandCount)
    {
      LOG(ERROR) << "Index out of range " << i << "/" << operandCount;
      return false;
    }
  }
  return true;
}

static bool validOperands(const std::vector<Operand> &operands, const std::vector<uint8_t> &operandValues)
{
  for (auto &operand : operands)
  {
    if (!validCode(kNumberOfDataTypes, static_cast<uint32_t>(operand.type)))
    {
      LOG(ERROR) << "Invalid operand type ";
      return false;
    }
    /* TODO validate dim with type
    if (!validOperandIndexes(operand.dimensions, mDimensions)) {
        return false;
    }
    */
    switch (operand.lifetime)
    {
      case OperandLifeTime::CONSTANT_COPY:
        if (operand.location.offset + operand.location.length > operandValues.size())
        {
          LOG(ERROR) << "OperandValue location out of range.  Starts at " << operand.location.offset
                     << ", length " << operand.location.length << ", max " << operandValues.size();
          return false;
        }
        break;
      case OperandLifeTime::TEMPORARY_VARIABLE:
      case OperandLifeTime::MODEL_INPUT:
      case OperandLifeTime::MODEL_OUTPUT:
      case OperandLifeTime::NO_VALUE:
        if (operand.location.offset != 0 || operand.location.length != 0)
        {
          LOG(ERROR) << "Unexpected offset " << operand.location.offset << " or length "
                     << operand.location.length << " for runtime location.";
          return false;
        }
        break;
      case OperandLifeTime::CONSTANT_REFERENCE:
#if 0
        if (operand.location.poolIndex >= poolCount)
        {
          LOG(ERROR) << "Invalid poolIndex " << operand.location.poolIndex << "/" << poolCount;
          return false;
        }
#endif
        break;
      // TODO: Validate that we are within the pool.
      default:
        LOG(ERROR) << "Invalid lifetime";
        return false;
    }
  }
  return true;
}

static bool validOperations(const std::vector<Operation> &operations, size_t operandCount)
{
  for (auto &op : operations)
  {
    if (!validCode(kNumberOfOperationTypes, static_cast<uint32_t>(op.type)))
    {
      LOG(ERROR) << "Invalid operation type ";
      return false;
    }
    if (!validOperandIndexes(op.inputs, operandCount) ||
        !validOperandIndexes(op.outputs, operandCount))
    {
      return false;
    }
  }
  return true;
}

// TODO doublecheck
bool validateModel(const Model &model)
{
  const size_t operandCount = model.operands.size();
  return (validOperands(model.operands, model.operandValues) &&
          validOperations(model.operations, operandCount) &&
          validOperandIndexes(model.inputIndexes, operandCount) &&
          validOperandIndexes(model.outputIndexes, operandCount));
}

bool validRequestArguments(const std::vector<RequestArgument> &arguments,
                           const std::vector<uint32_t> &operandIndexes,
                           const std::vector<Operand> &operands, size_t poolCount, const char *type)
{
  const size_t argumentCount = arguments.size();
  if (argumentCount != operandIndexes.size())
  {
    LOG(ERROR) << "Request specifies " << argumentCount << " " << type << "s but the model has "
               << operandIndexes.size();
    return false;
  }
  for (size_t argumentIndex = 0; argumentIndex < argumentCount; argumentIndex++)
  {
    const RequestArgument &argument = arguments[argumentIndex];
    const uint32_t operandIndex = operandIndexes[argumentIndex];
    const Operand &operand = operands[operandIndex];
    if (argument.hasNoValue)
    {
      if (argument.location.poolIndex != 0 || argument.location.offset != 0 ||
          argument.location.length != 0 || argument.dimensions.size() != 0)
      {
        LOG(ERROR) << "Request " << type << " " << argumentIndex
                   << " has no value yet has details.";
        return false;
      }
    }
    if (argument.location.poolIndex >= poolCount)
    {
      LOG(ERROR) << "Request " << type << " " << argumentIndex << " has an invalid poolIndex "
                 << argument.location.poolIndex << "/" << poolCount;
      return false;
    }
    // TODO: Validate that we are within the pool.
    uint32_t rank = argument.dimensions.size();
    if (rank > 0)
    {
      if (rank != operand.dimensions.size())
      {
        LOG(ERROR) << "Request " << type << " " << argumentIndex << " has number of dimensions ("
                   << rank << ") different than the model's (" << operand.dimensions.size() << ")";
        return false;
      }
      for (size_t i = 0; i < rank; i++)
      {
        if (argument.dimensions[i] != operand.dimensions[i] && operand.dimensions[i] != 0)
        {
          LOG(ERROR) << "Request " << type << " " << argumentIndex << " has dimension " << i
                     << " of " << operand.dimensions[i] << " different than the model's "
                     << operand.dimensions[i];
          return false;
        }
        if (argument.dimensions[i] == 0)
        {
          LOG(ERROR) << "Request " << type << " " << argumentIndex << " has dimension " << i
                     << " of zero";
          return false;
        }
      }
    }
  }
  return true;
}

// TODO doublecheck
bool validateRequest(const Request &request, const Model &model)
{
  //const size_t poolCount = request.pools.size();
  const size_t poolCount = 0;
  return (validRequestArguments(request.inputs, model.inputIndexes, model.operands, poolCount,
                                "input") &&
          validRequestArguments(request.outputs, model.outputIndexes, model.operands, poolCount,
                                "output"));
}
