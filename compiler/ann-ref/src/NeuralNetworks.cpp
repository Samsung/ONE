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

#include "NeuralNetworks.h"

#include "CompilationBuilder.h"
#include "ExecutionBuilder.h"
#include "ModelBuilder.h"
#include "Memory.h"

#include "Logging.h"

#include <memory>

int ANeuralNetworksMemory_createFromFd(size_t size, int prot, int fd, size_t offset,
                                       ANeuralNetworksMemory **memory)
{
  *memory = nullptr;
  auto m = std::make_unique<MappedMemory>();
  if (m == nullptr)
  {
    return ANEURALNETWORKS_OUT_OF_MEMORY;
  }
  int n = m->set(size, prot, fd, offset);
  if (n != ANEURALNETWORKS_NO_ERROR)
  {
    return n;
  }
  *memory = reinterpret_cast<ANeuralNetworksMemory *>(m.release());
  return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksMemory_free(ANeuralNetworksMemory *memory)
{
  // No validation.  Free of nullptr is valid.
  Memory *m = reinterpret_cast<Memory *>(memory);
  delete m;
}

int ANeuralNetworksModel_create(ANeuralNetworksModel **model)
{
  if (!model)
  {
    LOG(ERROR) << "ANeuralNetworksModel_create passed a nullptr";
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }
  ModelBuilder *m = new ModelBuilder();
  if (m == nullptr)
  {
    *model = nullptr;
    return ANEURALNETWORKS_OUT_OF_MEMORY;
  }
  *model = reinterpret_cast<ANeuralNetworksModel *>(m);
  return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksModel_free(ANeuralNetworksModel *model)
{
  // No validation.  Free of nullptr is valid.
  ModelBuilder *m = reinterpret_cast<ModelBuilder *>(model);
  delete m;
}

int ANeuralNetworksModel_finish(ANeuralNetworksModel *model)
{
  if (!model)
  {
    LOG(ERROR) << "ANeuralNetworksModel_finish passed a nullptr";
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }
  ModelBuilder *m = reinterpret_cast<ModelBuilder *>(model);
  return m->finish();
}

int ANeuralNetworksModel_addOperand(ANeuralNetworksModel *model,
                                    const ANeuralNetworksOperandType *type)
{
  if (!model || !type)
  {
    LOG(ERROR) << "ANeuralNetworksModel_addOperand passed a nullptr";
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }
  ModelBuilder *m = reinterpret_cast<ModelBuilder *>(model);
  return m->addOperand(*type);
}

int ANeuralNetworksModel_setOperandValue(ANeuralNetworksModel *model, int32_t index,
                                         const void *buffer, size_t length)
{
  if (!model || !buffer)
  {
    LOG(ERROR) << "ANeuralNetworksModel_setOperandValue passed a nullptr";
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }
  ModelBuilder *m = reinterpret_cast<ModelBuilder *>(model);
  return m->setOperandValue(index, buffer, length);
}

int ANeuralNetworksModel_setOperandValueFromMemory(ANeuralNetworksModel *model, int32_t index,
                                                   const ANeuralNetworksMemory *memory,
                                                   size_t offset, size_t length)
{
  if (!model || !memory)
  {
    LOG(ERROR) << "ANeuralNetworksModel_setOperandValue passed a nullptr";
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }
  const Memory *mem = reinterpret_cast<const Memory *>(memory);
  ModelBuilder *m = reinterpret_cast<ModelBuilder *>(model);
  return m->setOperandValueFromMemory(index, mem, offset, length);
}

int ANeuralNetworksModel_addOperation(ANeuralNetworksModel *model,
                                      ANeuralNetworksOperationType type, uint32_t inputCount,
                                      const uint32_t *inputs, uint32_t outputCount,
                                      const uint32_t *outputs)
{
  if (!model || !inputs || !outputs)
  {
    LOG(ERROR) << "ANeuralNetworksModel_addOperation passed a nullptr";
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }
  ModelBuilder *m = reinterpret_cast<ModelBuilder *>(model);
  return m->addOperation(static_cast<OperationType>(type), inputCount, inputs, outputCount,
                         outputs);
}

int ANeuralNetworksModel_identifyInputsAndOutputs(ANeuralNetworksModel *model, uint32_t inputCount,
                                                  const uint32_t *inputs, uint32_t outputCount,
                                                  const uint32_t *outputs)
{
  if (!model || !inputs || !outputs)
  {
    LOG(ERROR) << ("ANeuralNetworksModel_identifyInputsAndOutputs passed a nullptr");
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }
  ModelBuilder *m = reinterpret_cast<ModelBuilder *>(model);
  return m->identifyInputsAndOutputs(inputCount, inputs, outputCount, outputs);
}

int ANeuralNetworksCompilation_create(ANeuralNetworksModel *model,
                                      ANeuralNetworksCompilation **compilation)
{
  if (!model || !compilation)
  {
    LOG(ERROR) << "ANeuralNetworksCompilation_create passed a nullptr";
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  ModelBuilder *m = reinterpret_cast<ModelBuilder *>(model);
  CompilationBuilder *c = nullptr;
  int result = m->createCompilation(&c);
  *compilation = reinterpret_cast<ANeuralNetworksCompilation *>(c);
  return result;
}

void ANeuralNetworksCompilation_free(ANeuralNetworksCompilation *compilation)
{
  // No validation.  Free of nullptr is valid.
  // TODO specification says that a compilation-in-flight can be deleted
  CompilationBuilder *c = reinterpret_cast<CompilationBuilder *>(compilation);
  delete c;
}

int ANeuralNetworksCompilation_setPreference(ANeuralNetworksCompilation *compilation,
                                             int32_t preference)
{
  if (!compilation)
  {
    LOG(ERROR) << "ANeuralNetworksCompilation_setPreference passed a nullptr";
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }
  // NOTE Ignore preference
  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksCompilation_finish(ANeuralNetworksCompilation *compilation)
{
  if (!compilation)
  {
    LOG(ERROR) << "ANeuralNetworksCompilation_finish passed a nullptr";
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }
  CompilationBuilder *c = reinterpret_cast<CompilationBuilder *>(compilation);
  return c->finish();
}

int ANeuralNetworksExecution_create(ANeuralNetworksCompilation *compilation,
                                    ANeuralNetworksExecution **execution)
{
  if (!compilation || !execution)
  {
    LOG(ERROR) << "ANeuralNetworksExecution_create passed a nullptr";
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  CompilationBuilder *c = reinterpret_cast<CompilationBuilder *>(compilation);
  ExecutionBuilder *r = nullptr;
  int result = c->createExecution(&r);
  *execution = reinterpret_cast<ANeuralNetworksExecution *>(r);
  return result;
}

void ANeuralNetworksExecution_free(ANeuralNetworksExecution *execution)
{
  // TODO specification says that an execution-in-flight can be deleted
  // No validation.  Free of nullptr is valid.
  ExecutionBuilder *r = reinterpret_cast<ExecutionBuilder *>(execution);
  delete r;
}

int ANeuralNetworksExecution_setInput(ANeuralNetworksExecution *execution, int32_t index,
                                      const ANeuralNetworksOperandType *type, const void *buffer,
                                      size_t length)
{
  if (!execution)
  {
    LOG(ERROR) << "ANeuralNetworksExecution_setInput passed execution with a nullptr";
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }
  if (!buffer && length != 0)
  {
    LOG(ERROR) << "ANeuralNetworksExecution_setInput passed buffer with a nullptr";
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  ExecutionBuilder *r = reinterpret_cast<ExecutionBuilder *>(execution);
  return r->setInput(index, type, buffer, length);
}

int ANeuralNetworksExecution_setInputFromMemory(ANeuralNetworksExecution *execution, int32_t index,
                                                const ANeuralNetworksOperandType *type,
                                                const ANeuralNetworksMemory *memory, size_t offset,
                                                size_t length)
{
  if (!execution || !memory)
  {
    LOG(ERROR) << "ANeuralNetworksExecution_setInputFromMemory passed a nullptr";
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  const Memory *m = reinterpret_cast<const Memory *>(memory);
  ExecutionBuilder *r = reinterpret_cast<ExecutionBuilder *>(execution);
  return r->setInputFromMemory(index, type, m, offset, length);
}

int ANeuralNetworksExecution_setOutput(ANeuralNetworksExecution *execution, int32_t index,
                                       const ANeuralNetworksOperandType *type, void *buffer,
                                       size_t length)
{
  if (!execution || !buffer)
  {
    LOG(ERROR) << "ANeuralNetworksExecution_setOutput passed a nullptr";
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }
  ExecutionBuilder *r = reinterpret_cast<ExecutionBuilder *>(execution);
  return r->setOutput(index, type, buffer, length);
}

int ANeuralNetworksExecution_setOutputFromMemory(ANeuralNetworksExecution *execution, int32_t index,
                                                 const ANeuralNetworksOperandType *type,
                                                 const ANeuralNetworksMemory *memory, size_t offset,
                                                 size_t length)
{
  if (!execution || !memory)
  {
    LOG(ERROR) << "ANeuralNetworksExecution_setOutputFromMemory passed a nullptr";
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  ExecutionBuilder *r = reinterpret_cast<ExecutionBuilder *>(execution);
  const Memory *m = reinterpret_cast<const Memory *>(memory);
  return r->setOutputFromMemory(index, type, m, offset, length);
}

int ANeuralNetworksExecution_startCompute(ANeuralNetworksExecution *execution,
                                          ANeuralNetworksEvent **event)
{
  if (!execution || !event)
  {
    LOG(ERROR) << "ANeuralNetworksExecution_startCompute passed a nullptr";
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }
  // TODO validate the rest

  ExecutionBuilder *r = reinterpret_cast<ExecutionBuilder *>(execution);

  // Dynamically allocate an sp to wrap an ExecutionCallback, seen in the NN
  // API as an abstract event object. The sp<ExecutionCallback> object is
  // returned when the execution has been successfully launched, otherwise a
  // nullptr is returned. The sp is used for ref-counting purposes. Without
  // it, the HIDL service could attempt to communicate with a dead callback
  // object.
  *event = nullptr;

  int n = r->startCompute();
  if (n != ANEURALNETWORKS_NO_ERROR)
  {
    return n;
  }
  *event = reinterpret_cast<ANeuralNetworksEvent *>(new int);
  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksEvent_wait(ANeuralNetworksEvent *event)
{
  if (event == nullptr)
  {
    LOG(ERROR) << "ANeuralNetworksEvent_wait passed a nullptr";
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksEvent_free(ANeuralNetworksEvent *event)
{
  // No validation.  Free of nullptr is valid.
  if (event)
  {
    int *e = reinterpret_cast<int *>(event);
    delete e;
  }
}
