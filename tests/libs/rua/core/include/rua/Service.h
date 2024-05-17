/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

/**
 * @file Service.h
 * @brief Core abstraction that RUA depends on.
 */
#ifndef __NNFW_RUA_SERVICE_H__
#define __NNFW_RUA_SERVICE_H__

#include "NeuralNetworks.h"

struct ANeuralNetworksMemory;
struct ANeuralNetworksEvent;

struct ANeuralNetworksModel;
struct ANeuralNetworksCompilation;
struct ANeuralNetworksExecution;

namespace rua
{

/**
 * @brief A wrapper for ANeuralNetworkMemory API
 */
struct MemoryService
{
  virtual ~MemoryService() = default;

  virtual int createFromFd(size_t size, int protect, int fd, size_t offset,
                           ANeuralNetworksMemory **memory) const = 0;

  virtual void free(ANeuralNetworksMemory *memory) const = 0;
};

/**
 * @brief A wrapper for ANeuralNetworkModel API
 */
struct ModelService
{
  virtual ~ModelService() = default;

  virtual int create(ANeuralNetworksModel **model) const = 0;

  virtual int addOperand(ANeuralNetworksModel *model,
                         const ANeuralNetworksOperandType *type) const = 0;

  virtual int setOperandValue(ANeuralNetworksModel *model, int32_t index, const void *buffer,
                              size_t length) const = 0;

  virtual int setOperandValueFromMemory(ANeuralNetworksModel *model, int32_t index,
                                        const ANeuralNetworksMemory *memory, size_t offset,
                                        size_t length) const = 0;

  virtual int addOperation(ANeuralNetworksModel *model, ANeuralNetworksOperationType type,
                           uint32_t inputCount, const uint32_t *inputs, uint32_t outputCount,
                           const uint32_t *outputs) const = 0;

  virtual int identifyInputsAndOutputs(ANeuralNetworksModel *model, uint32_t inputCount,
                                       const uint32_t *inputs, uint32_t outputCount,
                                       const uint32_t *outputs) const = 0;

  virtual int relaxComputationFloat32toFloat16(ANeuralNetworksModel *model, bool allow) const = 0;

  virtual int finish(ANeuralNetworksModel *model) const = 0;

  virtual void free(ANeuralNetworksModel *model) const = 0;
};

/**
 * @brief A wrapper for ANeuralNetworkCompilation API
 */
struct CompilationService
{
  virtual ~CompilationService() = default;

  virtual int create(ANeuralNetworksModel *model,
                     ANeuralNetworksCompilation **compilation) const = 0;

  virtual int setPreference(ANeuralNetworksCompilation *compilation, int32_t preference) const = 0;
  virtual int finish(ANeuralNetworksCompilation *compilation) const = 0;

  virtual void free(ANeuralNetworksCompilation *compilation) const = 0;
};

/**
 * @brief A wrapper for ANeuralNetworkExecution API
 */
struct ExecutionService
{
  virtual ~ExecutionService() = default;

  virtual int create(ANeuralNetworksCompilation *compilation,
                     ANeuralNetworksExecution **execution) const = 0;

  virtual void free(ANeuralNetworksExecution *execution) const = 0;

  virtual int setInput(ANeuralNetworksExecution *execution, int32_t index,
                       const ANeuralNetworksOperandType *type, const void *buffer,
                       size_t length) const = 0;

  virtual int setInputFromMemory(ANeuralNetworksExecution *execution, int32_t index,
                                 const ANeuralNetworksOperandType *type,
                                 const ANeuralNetworksMemory *memory, size_t offset,
                                 size_t length) const = 0;

  virtual int setOutput(ANeuralNetworksExecution *execution, int32_t index,
                        const ANeuralNetworksOperandType *type, void *buffer,
                        size_t length) const = 0;

  virtual int setOutputFromMemory(ANeuralNetworksExecution *execution, int32_t index,
                                  const ANeuralNetworksOperandType *type,
                                  const ANeuralNetworksMemory *memory, size_t offset,
                                  size_t length) const = 0;

  virtual int startCompute(ANeuralNetworksExecution *execution,
                           ANeuralNetworksEvent **event) const = 0;
};

/**
 * @brief A wrapper for ANeuralNetworkEvent API
 */
struct EventService
{
  virtual int wait(ANeuralNetworksEvent *event) const = 0;
  virtual void free(ANeuralNetworksEvent *event) const = 0;
};

/**
 * @brief A wrapper for Android NN rutime itself
 */
struct RuntimeService
{
  virtual ~RuntimeService() = default;

  virtual const MemoryService *memory(void) const = 0;
  virtual const ModelService *model(void) const = 0;
  virtual const CompilationService *compilation(void) const = 0;
  virtual const ExecutionService *execution(void) const = 0;
  virtual const EventService *event(void) const = 0;
};

} // namespace rua

#endif // __NNFW_RUA_SERVICE_H__
