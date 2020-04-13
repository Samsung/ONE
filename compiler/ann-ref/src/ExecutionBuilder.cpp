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

#include "ExecutionBuilder.h"
#include "CompilationBuilder.h"
#include "ModelBuilder.h"

#include "Executor.h"

#include "Logging.h"
#include "Validation.h"

static void setRequestArgumentArray(const std::vector<ModelArgumentInfo> &argumentInfos,
                                    std::vector<RequestArgument> *ioInfos)
{
  size_t count = argumentInfos.size();
  ioInfos->resize(count);
  for (size_t i = 0; i < count; i++)
  {
    const auto &info = argumentInfos[i];
    (*ioInfos)[i] = {
        .hasNoValue = info.state == ModelArgumentInfo::HAS_NO_VALUE,
        .location = info.locationAndLength,
        .dimensions = info.dimensions,
    };
  }
}

bool setRunTimePoolInfosFromMemories(std::vector<RunTimePoolInfo> *poolInfos,
                                     const std::vector<uint8_t *> &pools)
{
  poolInfos->resize(pools.size());
  for (size_t i = 0; i < pools.size(); i++)
  {
    auto &poolInfo = (*poolInfos)[i];
    if (!poolInfo.set(pools[i]))
    {
      LOG(ERROR) << "Could not map pool";
      return false;
    }
  }
  return true;
}

ExecutionBuilder::ExecutionBuilder(const ModelBuilder *model)
    : mModel(model), mInputs(mModel->inputCount()), mOutputs(mModel->outputCount())
{
  VLOG(EXECUTION) << "ExecutionBuilder::ExecutionBuilder";
}

int ExecutionBuilder::setInput(uint32_t index, const ANeuralNetworksOperandType *type,
                               const void *buffer, size_t length)
{
  uint32_t count = static_cast<uint32_t>(mInputs.size());
  if (index >= count)
  {
    LOG(ERROR) << "ANeuralNetworksExecution_setInput bad index " << index << " " << count;
    return ANEURALNETWORKS_BAD_DATA;
  }
  if (type != nullptr)
  {
    int n = validateOperandType(*type, "ANeuralNetworksExecution_setInput", false);
    if (n != ANEURALNETWORKS_NO_ERROR)
    {
      return n;
    }
  }
  if (length > 0xFFFFFFFF)
  {
    LOG(ERROR) << "ANeuralNetworksExecution_setInput input exceeds max length " << length;
    return ANEURALNETWORKS_BAD_DATA;
  }
  uint32_t l = static_cast<uint32_t>(length);
  return mInputs[index].setFromPointer(mModel->getInputOperand(index), type,
                                       const_cast<void *>(buffer), l);
}

int ExecutionBuilder::setInputFromMemory(uint32_t index, const ANeuralNetworksOperandType *type,
                                         const Memory *memory, size_t offset, size_t length)
{
  uint32_t count = static_cast<uint32_t>(mInputs.size());
  if (index >= count)
  {
    LOG(ERROR) << "ANeuralNetworksExecution_setInputFromMemory bad index " << index << " " << count;
    return ANEURALNETWORKS_BAD_DATA;
  }
  if (!memory->validateSize(offset, length))
  {
    return ANEURALNETWORKS_BAD_DATA;
  }
  uint32_t poolIndex = mMemories.add(memory);
  return mInputs[index].setFromMemory(mModel->getInputOperand(index), type, poolIndex, offset,
                                      length);
}

int ExecutionBuilder::setOutput(uint32_t index, const ANeuralNetworksOperandType *type,
                                void *buffer, size_t length)
{
  uint32_t count = static_cast<uint32_t>(mOutputs.size());
  if (index >= count)
  {
    LOG(ERROR) << "ANeuralNetworksExecution_setOutput bad index " << index << " " << count;
    return ANEURALNETWORKS_BAD_DATA;
  }
  if (type != nullptr)
  {
    int n = validateOperandType(*type, "ANeuralNetworksExecution_setOutput", false);
    if (n != ANEURALNETWORKS_NO_ERROR)
    {
      return n;
    }
  }
  if (length > 0xFFFFFFFF)
  {
    LOG(ERROR) << "ANeuralNetworksExecution_setOutput input exceeds max length " << length;
    return ANEURALNETWORKS_BAD_DATA;
  }
  uint32_t l = static_cast<uint32_t>(length);
  return mOutputs[index].setFromPointer(mModel->getOutputOperand(index), type, buffer, l);
}

int ExecutionBuilder::setOutputFromMemory(uint32_t index, const ANeuralNetworksOperandType *type,
                                          const Memory *memory, size_t offset, size_t length)
{
  // Should be similar to StepExecutor::setInputOrOutputFromTemporaryMemory()

  uint32_t count = static_cast<uint32_t>(mOutputs.size());
  if (index >= count)
  {
    LOG(ERROR) << "ANeuralNetworksExecution_setOutputFromMemory bad index " << index << " "
               << count;
    return ANEURALNETWORKS_BAD_DATA;
  }
  if (!memory->validateSize(offset, length))
  {
    return ANEURALNETWORKS_BAD_DATA;
  }
  // TODO validate the rest
  uint32_t poolIndex = mMemories.add(memory);
  return mOutputs[index].setFromMemory(mModel->getOutputOperand(index), type, poolIndex, offset,
                                       length);
}

int ExecutionBuilder::startCompute(void)
{
  Model model;
  mModel->publish(&model);

  // modelPoolInfo holds the infomation of pre-allocated memory pools during model construction
  std::vector<RunTimePoolInfo> modelPoolInfos;
  if (!setRunTimePoolInfosFromMemories(&modelPoolInfos, model.pools))
  {
    return ANEURALNETWORKS_UNMAPPABLE;
  }

  std::vector<RunTimePoolInfo> requestPoolInfos;
  uint32_t count = mMemories.size();
  requestPoolInfos.resize(count);
  // Create as many pools as there are input / output
  auto fixPointerArguments = [&requestPoolInfos](std::vector<ModelArgumentInfo> &argumentInfos) {
    for (ModelArgumentInfo &argumentInfo : argumentInfos)
    {
      if (argumentInfo.state == ModelArgumentInfo::POINTER)
      {
        RunTimePoolInfo runTimeInfo;
        runTimeInfo.buffer = static_cast<uint8_t *>(argumentInfo.buffer);
        argumentInfo.locationAndLength.poolIndex = static_cast<uint32_t>(requestPoolInfos.size());
        argumentInfo.locationAndLength.offset = 0;
        requestPoolInfos.push_back(runTimeInfo);
      }
    }
  };
  fixPointerArguments(mInputs);
  fixPointerArguments(mOutputs);

  Request request;
  setRequestArgumentArray(mInputs, &request.inputs);
  setRequestArgumentArray(mOutputs, &request.outputs);

  Executor executor;
  return executor.run(model, request, modelPoolInfos, requestPoolInfos);
}
