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

#include "rua/DynamicBinder.h"

#include "NeuralNetworksLoadHelpers.h"

using namespace rua;

//
// Memory
//
namespace
{

typedef int (*ANeuralNetworksMemory_createFromFd_fn)(size_t size, int protect, int fd,
                                                     size_t offset, ANeuralNetworksMemory **memory);

typedef void (*ANeuralNetworksMemory_free_fn)(ANeuralNetworksMemory *memory);

struct MemoryServiceImpl final : public MemoryService
{
  int createFromFd(size_t size, int protect, int fd, size_t offset,
                   ANeuralNetworksMemory **memory) const override
  {
    LOAD_FUNCTION(ANeuralNetworksMemory_createFromFd);
    EXECUTE_FUNCTION_RETURN(size, protect, fd, offset, memory);
  }

  void free(ANeuralNetworksMemory *memory) const override
  {
    LOAD_FUNCTION(ANeuralNetworksMemory_free);
    EXECUTE_FUNCTION(memory);
  }
};

} // namespace

//
// Event
//
namespace
{

typedef int (*ANeuralNetworksEvent_wait_fn)(ANeuralNetworksEvent *event);

typedef void (*ANeuralNetworksEvent_free_fn)(ANeuralNetworksEvent *event);

struct EventServiceImpl final : public EventService
{

  int wait(ANeuralNetworksEvent *event) const override
  {
    LOAD_FUNCTION(ANeuralNetworksEvent_wait);
    EXECUTE_FUNCTION_RETURN(event);
  }

  void free(ANeuralNetworksEvent *event) const override
  {
    LOAD_FUNCTION(ANeuralNetworksEvent_free);
    EXECUTE_FUNCTION(event);
  }
};

} // namespace

//
// Model
//
namespace
{

typedef int (*ANeuralNetworksModel_create_fn)(ANeuralNetworksModel **model);

typedef int (*ANeuralNetworksModel_finish_fn)(ANeuralNetworksModel *model);

typedef void (*ANeuralNetworksModel_free_fn)(ANeuralNetworksModel *model);

typedef int (*ANeuralNetworksModel_addOperand_fn)(ANeuralNetworksModel *model,
                                                  const ANeuralNetworksOperandType *type);

typedef int (*ANeuralNetworksModel_setOperandValue_fn)(ANeuralNetworksModel *model, int32_t index,
                                                       const void *buffer, size_t length);

typedef int (*ANeuralNetworksModel_setOperandValueFromMemory_fn)(
  ANeuralNetworksModel *model, int32_t index, const ANeuralNetworksMemory *memory, size_t offset,
  size_t length);

typedef int (*ANeuralNetworksModel_addOperation_fn)(ANeuralNetworksModel *model,
                                                    ANeuralNetworksOperationType type,
                                                    uint32_t inputCount, const uint32_t *inputs,
                                                    uint32_t outputCount, const uint32_t *outputs);

typedef int (*ANeuralNetworksModel_identifyInputsAndOutputs_fn)(ANeuralNetworksModel *model,
                                                                uint32_t inputCount,
                                                                const uint32_t *inputs,
                                                                uint32_t outputCount,
                                                                const uint32_t *outputs);

typedef int (*ANeuralNetworksModel_relaxComputationFloat32toFloat16_fn)(ANeuralNetworksModel *model,
                                                                        bool allow);

struct ModelServiceImpl final : public ModelService
{
  int create(ANeuralNetworksModel **model) const override
  {
    LOAD_FUNCTION(ANeuralNetworksModel_create);
    EXECUTE_FUNCTION_RETURN(model);
  }

  int addOperand(ANeuralNetworksModel *model, const ANeuralNetworksOperandType *type) const override
  {
    LOAD_FUNCTION(ANeuralNetworksModel_addOperand);
    EXECUTE_FUNCTION_RETURN(model, type);
  }
  int setOperandValue(ANeuralNetworksModel *model, int32_t index, const void *buffer,
                      size_t length) const override
  {
    LOAD_FUNCTION(ANeuralNetworksModel_setOperandValue);
    EXECUTE_FUNCTION_RETURN(model, index, buffer, length);
  }

  int setOperandValueFromMemory(ANeuralNetworksModel *model, int32_t index,
                                const ANeuralNetworksMemory *memory, size_t offset,
                                size_t length) const override
  {
    LOAD_FUNCTION(ANeuralNetworksModel_setOperandValueFromMemory);
    EXECUTE_FUNCTION_RETURN(model, index, memory, offset, length);
  }

  int addOperation(ANeuralNetworksModel *model, ANeuralNetworksOperationType type,
                   uint32_t inputCount, const uint32_t *inputs, uint32_t outputCount,
                   const uint32_t *outputs) const override
  {
    LOAD_FUNCTION(ANeuralNetworksModel_addOperation);
    EXECUTE_FUNCTION_RETURN(model, type, inputCount, inputs, outputCount, outputs);
  }

  int identifyInputsAndOutputs(ANeuralNetworksModel *model, uint32_t inputCount,
                               const uint32_t *inputs, uint32_t outputCount,
                               const uint32_t *outputs) const override
  {
    LOAD_FUNCTION(ANeuralNetworksModel_identifyInputsAndOutputs);
    EXECUTE_FUNCTION_RETURN(model, inputCount, inputs, outputCount, outputs);
  }

  int relaxComputationFloat32toFloat16(ANeuralNetworksModel *model, bool allow) const override
  {
    LOAD_FUNCTION(ANeuralNetworksModel_relaxComputationFloat32toFloat16);
    EXECUTE_FUNCTION_RETURN(model, allow);
  }

  int finish(ANeuralNetworksModel *model) const override
  {
    LOAD_FUNCTION(ANeuralNetworksModel_finish);
    EXECUTE_FUNCTION_RETURN(model);
  }

  void free(ANeuralNetworksModel *model) const override
  {
    LOAD_FUNCTION(ANeuralNetworksModel_free);
    EXECUTE_FUNCTION(model);
  }
};

} // namespace

//
// Compilation
//
namespace
{

typedef int (*ANeuralNetworksCompilation_create_fn)(ANeuralNetworksModel *model,
                                                    ANeuralNetworksCompilation **compilation);

typedef void (*ANeuralNetworksCompilation_free_fn)(ANeuralNetworksCompilation *compilation);

typedef int (*ANeuralNetworksCompilation_setPreference_fn)(ANeuralNetworksCompilation *compilation,
                                                           int32_t preference);

typedef int (*ANeuralNetworksCompilation_finish_fn)(ANeuralNetworksCompilation *compilation);

struct CompilationServiceImpl : public CompilationService
{

  int create(ANeuralNetworksModel *model, ANeuralNetworksCompilation **compilation) const override
  {
    LOAD_FUNCTION(ANeuralNetworksCompilation_create);
    EXECUTE_FUNCTION_RETURN(model, compilation);
  }

  int setPreference(ANeuralNetworksCompilation *compilation, int32_t preference) const override
  {
    LOAD_FUNCTION(ANeuralNetworksCompilation_setPreference);
    EXECUTE_FUNCTION_RETURN(compilation, preference);
  }

  int finish(ANeuralNetworksCompilation *compilation) const override
  {
    LOAD_FUNCTION(ANeuralNetworksCompilation_finish);
    EXECUTE_FUNCTION_RETURN(compilation);
  }

  void free(ANeuralNetworksCompilation *compilation) const override
  {
    LOAD_FUNCTION(ANeuralNetworksCompilation_free);
    EXECUTE_FUNCTION(compilation);
  }
};

} // namespace

//
// Exceution
//
namespace
{

typedef int (*ANeuralNetworksExecution_create_fn)(ANeuralNetworksCompilation *compilation,
                                                  ANeuralNetworksExecution **execution);

typedef void (*ANeuralNetworksExecution_free_fn)(ANeuralNetworksExecution *execution);

typedef int (*ANeuralNetworksExecution_setInput_fn)(ANeuralNetworksExecution *execution,
                                                    int32_t index,
                                                    const ANeuralNetworksOperandType *type,
                                                    const void *buffer, size_t length);

typedef int (*ANeuralNetworksExecution_setInputFromMemory_fn)(
  ANeuralNetworksExecution *execution, int32_t index, const ANeuralNetworksOperandType *type,
  const ANeuralNetworksMemory *memory, size_t offset, size_t length);

typedef int (*ANeuralNetworksExecution_setOutput_fn)(ANeuralNetworksExecution *execution,
                                                     int32_t index,
                                                     const ANeuralNetworksOperandType *type,
                                                     void *buffer, size_t length);

typedef int (*ANeuralNetworksExecution_setOutputFromMemory_fn)(
  ANeuralNetworksExecution *execution, int32_t index, const ANeuralNetworksOperandType *type,
  const ANeuralNetworksMemory *memory, size_t offset, size_t length);

typedef int (*ANeuralNetworksExecution_startCompute_fn)(ANeuralNetworksExecution *execution,
                                                        ANeuralNetworksEvent **event);

struct ExecutionServiceImpl : public ExecutionService
{

  int create(ANeuralNetworksCompilation *compilation,
             ANeuralNetworksExecution **execution) const override
  {
    LOAD_FUNCTION(ANeuralNetworksExecution_create);
    EXECUTE_FUNCTION_RETURN(compilation, execution);
  }

  int setInput(ANeuralNetworksExecution *execution, int32_t index,
               const ANeuralNetworksOperandType *type, const void *buffer,
               size_t length) const override
  {
    LOAD_FUNCTION(ANeuralNetworksExecution_setInput);
    EXECUTE_FUNCTION_RETURN(execution, index, type, buffer, length);
  }

  int setInputFromMemory(ANeuralNetworksExecution *execution, int32_t index,
                         const ANeuralNetworksOperandType *type,
                         const ANeuralNetworksMemory *memory, size_t offset,
                         size_t length) const override
  {
    LOAD_FUNCTION(ANeuralNetworksExecution_setInputFromMemory);
    EXECUTE_FUNCTION_RETURN(execution, index, type, memory, offset, length);
  }

  int setOutput(ANeuralNetworksExecution *execution, int32_t index,
                const ANeuralNetworksOperandType *type, void *buffer, size_t length) const override
  {
    LOAD_FUNCTION(ANeuralNetworksExecution_setOutput);
    EXECUTE_FUNCTION_RETURN(execution, index, type, buffer, length);
  }

  int setOutputFromMemory(ANeuralNetworksExecution *execution, int32_t index,
                          const ANeuralNetworksOperandType *type,
                          const ANeuralNetworksMemory *memory, size_t offset,
                          size_t length) const override
  {
    LOAD_FUNCTION(ANeuralNetworksExecution_setOutputFromMemory);
    EXECUTE_FUNCTION_RETURN(execution, index, type, memory, offset, length);
  }

  int startCompute(ANeuralNetworksExecution *execution, ANeuralNetworksEvent **event) const override
  {
    LOAD_FUNCTION(ANeuralNetworksExecution_startCompute);
    EXECUTE_FUNCTION_RETURN(execution, event);
  }

  void free(ANeuralNetworksExecution *execution) const override
  {
    LOAD_FUNCTION(ANeuralNetworksExecution_free);
    EXECUTE_FUNCTION(execution);
  }
};

} // namespace

//
// Runtime
//
namespace
{

class RuntimeImpl final : public RuntimeService
{
public:
  const MemoryService *memory(void) const override { return &_memory; }
  const EventService *event(void) const override { return &_event; }

  const ModelService *model(void) const override { return &_model; }
  const CompilationService *compilation(void) const override { return &_compilation; }
  const ExecutionService *execution(void) const override { return &_execution; }

private:
  MemoryServiceImpl _memory;
  EventServiceImpl _event;

  ModelServiceImpl _model;
  CompilationServiceImpl _compilation;
  ExecutionServiceImpl _execution;
};

} // namespace

namespace rua
{

const RuntimeService *DynamicBinder::get(void)
{
  static RuntimeImpl runtime;
  return &runtime;
}

} // namespace rua
