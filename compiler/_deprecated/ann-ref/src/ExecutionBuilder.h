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

#ifndef __EXECUTION_BUILDER_H__
#define __EXECUTION_BUILDER_H__

#include "NeuralNetworks.h"

#include "ModelBuilder.h"
#include "ModelArgumentInfo.h"

#include "Memory.h"

#include <vector>

class ModelBuilder;

class ExecutionBuilder
{
public:
  ExecutionBuilder(const ModelBuilder *);

public:
  int setInput(uint32_t index, const ANeuralNetworksOperandType *type, const void *buffer,
               size_t length);
  int setInputFromMemory(uint32_t index, const ANeuralNetworksOperandType *type,
                         const Memory *memory, size_t offset, size_t length);

public:
  int setOutput(uint32_t index, const ANeuralNetworksOperandType *type, void *buffer,
                size_t length);
  int setOutputFromMemory(uint32_t index, const ANeuralNetworksOperandType *type,
                          const Memory *memory, size_t offset, size_t length);

public:
  int startCompute(void);

private:
  const ModelBuilder *mModel;

private:
  // The information we'll send to the driver about the inputs and outputs.
  // Note that we build this in two steps:
  // 1. As the arguments are specified, set the corresponding mInputs or mOutputs element.
  //    If set from a pointer, don't set the location in the RequestArgument but store it
  //    instead in mInputBuffers or mOutputBuffers.
  // 2. Once we have all the inputs and outputs, if needed, allocate shared memory for
  //    the m*Buffers entries.  Copy the input values into the shared memory.
  // We do this to avoid creating a lot of shared memory objects if we have a lot of
  // parameters specified via pointers.  We also avoid copying in the case where
  // some of the nodes will interpreted on the CPU anyway.
  std::vector<ModelArgumentInfo> mInputs;
  std::vector<ModelArgumentInfo> mOutputs;

private:
  MemoryTracker mMemories;
};

#endif // __EXECUTION_BUILDER_H__
