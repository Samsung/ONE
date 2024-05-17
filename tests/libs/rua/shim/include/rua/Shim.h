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

#ifndef __NNFW_RUA_SHIM_H__
#define __NNFW_RUA_SHIM_H__

#include <rua/Anchor.h>

//
// Memory
//
inline int ANeuralNetworksMemory_createFromFd(size_t size, int protect, int fd, size_t offset,
                                              ANeuralNetworksMemory **memory)
{
  return rua::Anchor::get()->memory()->createFromFd(size, protect, fd, offset, memory);
}

inline void ANeuralNetworksMemory_free(ANeuralNetworksMemory *memory)
{
  return rua::Anchor::get()->memory()->free(memory);
}

//
// Event
//
inline int ANeuralNetworksEvent_wait(ANeuralNetworksEvent *event)
{
  return rua::Anchor::get()->event()->wait(event);
}

inline void ANeuralNetworksEvent_free(ANeuralNetworksEvent *event)
{
  return rua::Anchor::get()->event()->free(event);
}

//
// Model
//
inline int ANeuralNetworksModel_create(ANeuralNetworksModel **model)
{
  return rua::Anchor::get()->model()->create(model);
}

inline int ANeuralNetworksModel_addOperand(ANeuralNetworksModel *model,
                                           const ANeuralNetworksOperandType *type)
{
  return rua::Anchor::get()->model()->addOperand(model, type);
}

inline int ANeuralNetworksModel_setOperandValue(ANeuralNetworksModel *model, int32_t index,
                                                const void *buffer, size_t length)
{
  return rua::Anchor::get()->model()->setOperandValue(model, index, buffer, length);
}

inline int ANeuralNetworksModel_setOperandValueFromMemory(ANeuralNetworksModel *model,
                                                          int32_t index,
                                                          const ANeuralNetworksMemory *memory,
                                                          size_t offset, size_t length)
{
  return rua::Anchor::get()->model()->setOperandValueFromMemory(model, index, memory, offset,
                                                                length);
}

inline int ANeuralNetworksModel_addOperation(ANeuralNetworksModel *model,
                                             ANeuralNetworksOperationType type, uint32_t inputCount,
                                             const uint32_t *inputs, uint32_t outputCount,
                                             const uint32_t *outputs)
{
  return rua::Anchor::get()->model()->addOperation(model, type, inputCount, inputs, outputCount,
                                                   outputs);
}

inline int ANeuralNetworksModel_identifyInputsAndOutputs(ANeuralNetworksModel *model,
                                                         uint32_t inputCount,
                                                         const uint32_t *inputs,
                                                         uint32_t outputCount,
                                                         const uint32_t *outputs)
{
  return rua::Anchor::get()->model()->identifyInputsAndOutputs(model, inputCount, inputs,
                                                               outputCount, outputs);
}

inline int ANeuralNetworksModel_relaxComputationFloat32toFloat16(ANeuralNetworksModel *model,
                                                                 bool allow)
{
  return rua::Anchor::get()->model()->relaxComputationFloat32toFloat16(model, allow);
}

inline int ANeuralNetworksModel_finish(ANeuralNetworksModel *model)
{
  return rua::Anchor::get()->model()->finish(model);
}

inline void ANeuralNetworksModel_free(ANeuralNetworksModel *model)
{
  return rua::Anchor::get()->model()->free(model);
}

//
// Compilation
//
inline int ANeuralNetworksCompilation_create(ANeuralNetworksModel *model,
                                             ANeuralNetworksCompilation **compilation)
{
  return rua::Anchor::get()->compilation()->create(model, compilation);
}

inline int ANeuralNetworksCompilation_setPreference(ANeuralNetworksCompilation *compilation,
                                                    int32_t preference)
{
  return rua::Anchor::get()->compilation()->setPreference(compilation, preference);
}

inline int ANeuralNetworksCompilation_finish(ANeuralNetworksCompilation *compilation)
{
  return rua::Anchor::get()->compilation()->finish(compilation);
}

inline void ANeuralNetworksCompilation_free(ANeuralNetworksCompilation *compilation)
{
  return rua::Anchor::get()->compilation()->free(compilation);
}

//
// Execution
//
inline int ANeuralNetworksExecution_create(ANeuralNetworksCompilation *compilation,
                                           ANeuralNetworksExecution **execution)
{
  return rua::Anchor::get()->execution()->create(compilation, execution);
}

inline int ANeuralNetworksExecution_setInput(ANeuralNetworksExecution *execution, int32_t index,
                                             const ANeuralNetworksOperandType *type,
                                             const void *buffer, size_t length)
{
  return rua::Anchor::get()->execution()->setInput(execution, index, type, buffer, length);
}

inline int ANeuralNetworksExecution_setInputFromMemory(ANeuralNetworksExecution *execution,
                                                       int32_t index,
                                                       const ANeuralNetworksOperandType *type,
                                                       const ANeuralNetworksMemory *memory,
                                                       size_t offset, size_t length)
{
  return rua::Anchor::get()->execution()->setInputFromMemory(execution, index, type, memory, offset,
                                                             length);
}

inline int ANeuralNetworksExecution_setOutput(ANeuralNetworksExecution *execution, int32_t index,
                                              const ANeuralNetworksOperandType *type, void *buffer,
                                              size_t length)
{
  return rua::Anchor::get()->execution()->setOutput(execution, index, type, buffer, length);
}

inline int ANeuralNetworksExecution_setOutputFromMemory(ANeuralNetworksExecution *execution,
                                                        int32_t index,
                                                        const ANeuralNetworksOperandType *type,
                                                        const ANeuralNetworksMemory *memory,
                                                        size_t offset, size_t length)
{
  return rua::Anchor::get()->execution()->setOutputFromMemory(execution, index, type, memory,
                                                              offset, length);
}

inline int ANeuralNetworksExecution_startCompute(ANeuralNetworksExecution *execution,
                                                 ANeuralNetworksEvent **event)
{
  return rua::Anchor::get()->execution()->startCompute(execution, event);
}

inline void ANeuralNetworksExecution_free(ANeuralNetworksExecution *execution)
{
  return rua::Anchor::get()->execution()->free(execution);
}

#endif // __NNFW_RUA_SHIM_H__
