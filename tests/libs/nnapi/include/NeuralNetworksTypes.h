/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// NOTE This header is derived from part of the following file
// https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/lite/nnapi/NeuralNetworksTypes.h

#ifndef __NEURAL_NETWORKS_TYPES_H__
#define __NEURAL_NETWORKS_TYPES_H__

#include "NeuralNetworks.h"

// NN api types based on NNAPI header file
// https://developer.android.com/ndk/reference/group/neural-networks

// nn api function types

typedef int (*ANeuralNetworksMemory_createFromFd_fn)(size_t size, int protect, int fd,
                                                     size_t offset, ANeuralNetworksMemory **memory);

typedef void (*ANeuralNetworksMemory_free_fn)(ANeuralNetworksMemory *memory);

typedef int (*ANeuralNetworksModel_create_fn)(ANeuralNetworksModel **model);

typedef int (*ANeuralNetworksModel_finish_fn)(ANeuralNetworksModel *model);

typedef void (*ANeuralNetworksModel_free_fn)(ANeuralNetworksModel *model);

typedef int (*ANeuralNetworksCompilation_create_fn)(ANeuralNetworksModel *model,
                                                    ANeuralNetworksCompilation **compilation);

typedef void (*ANeuralNetworksCompilation_free_fn)(ANeuralNetworksCompilation *compilation);

typedef int (*ANeuralNetworksCompilation_setPreference_fn)(ANeuralNetworksCompilation *compilation,
                                                           int32_t preference);

typedef int (*ANeuralNetworksCompilation_finish_fn)(ANeuralNetworksCompilation *compilation);

typedef int (*ANeuralNetworksModel_addOperand_fn)(ANeuralNetworksModel *model,
                                                  const ANeuralNetworksOperandType *type);

typedef int (*ANeuralNetworksModel_setOperandValue_fn)(ANeuralNetworksModel *model, int32_t index,
                                                       const void *buffer, size_t length);

typedef int (*ANeuralNetworksModel_setOperandSymmPerChannelQuantParams_fn)(
  ANeuralNetworksModel *model, int32_t index,
  const ANeuralNetworksSymmPerChannelQuantParams *channelQuant);

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

typedef int (*ANeuralNetworksEvent_wait_fn)(ANeuralNetworksEvent *event);

typedef void (*ANeuralNetworksEvent_free_fn)(ANeuralNetworksEvent *event);

typedef int (*ASharedMemory_create_fn)(const char *name, size_t size);

typedef int (*ANeuralNetworks_getDeviceCount_fn)(uint32_t *numDevices);

typedef int (*ANeuralNetworks_getDevice_fn)(uint32_t devIndex, ANeuralNetworksDevice **device);

typedef int (*ANeuralNetworksDevice_getName_fn)(const ANeuralNetworksDevice *device,
                                                const char **name);

typedef int (*ANeuralNetworksDevice_getType_fn)(const ANeuralNetworksDevice *device, int32_t *type);

typedef int (*ANeuralNetworksDevice_getVersion_fn)(const ANeuralNetworksDevice *device,
                                                   const char **version);

typedef int (*ANeuralNetworksDevice_getFeatureLevel_fn)(const ANeuralNetworksDevice *device,
                                                        int64_t *featureLevel);

typedef int (*ANeuralNetworksModel_getSupportedOperationsForDevices_fn)(
  const ANeuralNetworksModel *model, const ANeuralNetworksDevice *const *devices,
  uint32_t numDevices, bool *supportedOps);

typedef int (*ANeuralNetworksCompilation_createForDevices_fn)(
  ANeuralNetworksModel *model, const ANeuralNetworksDevice *const *devices, uint32_t numDevices,
  ANeuralNetworksCompilation **compilation);

typedef int (*ANeuralNetworksCompilation_setCaching_fn)(ANeuralNetworksCompilation *compilation,
                                                        const char *cacheDir, const uint8_t *token);

#if __ANDROID_API__ >= 30
typedef int (*ANeuralNetworksCompilation_setTimeout_fn)(ANeuralNetworksCompilation *compilation,
                                                        uint64_t duration);

typedef int (*ANeuralNetworksCompilation_setPriority_fn)(ANeuralNetworksCompilation *compilation,
                                                         int priority);
#endif // __ANDROID_API__ >= 30

typedef int (*ANeuralNetworksExecution_compute_fn)(ANeuralNetworksExecution *execution);

#if __ANDROID_API__ >= 30
typedef int (*ANeuralNetworksExecution_setTimeout_fn)(ANeuralNetworksExecution *execution,
                                                      uint64_t duration);

typedef int (*ANeuralNetworksExecution_setLoopTimeout_fn)(ANeuralNetworksExecution *execution,
                                                          uint64_t duration);
#endif // __ANDROID_API__ >= 30

typedef int (*ANeuralNetworksExecution_getOutputOperandRank_fn)(ANeuralNetworksExecution *execution,
                                                                int32_t index, uint32_t *rank);

typedef int (*ANeuralNetworksExecution_getOutputOperandDimensions_fn)(
  ANeuralNetworksExecution *execution, int32_t index, uint32_t *dimensions);

typedef int (*ANeuralNetworksBurst_create_fn)(ANeuralNetworksCompilation *compilation,
                                              ANeuralNetworksBurst **burst);

typedef void (*ANeuralNetworksBurst_free_fn)(ANeuralNetworksBurst *burst);

typedef int (*ANeuralNetworksExecution_burstCompute_fn)(ANeuralNetworksExecution *execution,
                                                        ANeuralNetworksBurst *burst);

typedef int (*ANeuralNetworksMemory_createFromAHardwareBuffer_fn)(const AHardwareBuffer *ahwb,
                                                                  ANeuralNetworksMemory **memory);

typedef int (*ANeuralNetworksExecution_setMeasureTiming_fn)(ANeuralNetworksExecution *execution,
                                                            bool measure);

typedef int (*ANeuralNetworksExecution_getDuration_fn)(const ANeuralNetworksExecution *execution,
                                                       int32_t durationCode, uint64_t *duration);

typedef int (*ANeuralNetworksDevice_getExtensionSupport_fn)(const ANeuralNetworksDevice *device,
                                                            const char *extensionName,
                                                            bool *isExtensionSupported);

typedef int (*ANeuralNetworksModel_getExtensionOperandType_fn)(ANeuralNetworksModel *model,
                                                               const char *extensionName,
                                                               uint16_t operandCodeWithinExtension,
                                                               int32_t *type);

typedef int (*ANeuralNetworksModel_getExtensionOperationType_fn)(
  ANeuralNetworksModel *model, const char *extensionName, uint16_t operationCodeWithinExtension,
  ANeuralNetworksOperationType *type);

typedef int (*ANeuralNetworksModel_setOperandExtensionData_fn)(ANeuralNetworksModel *model,
                                                               int32_t index, const void *data,
                                                               size_t length);

#if __ANDROID_API__ >= 30
typedef int (*ANeuralNetworksMemoryDesc_create_fn)(ANeuralNetworksMemoryDesc **desc);

typedef void (*ANeuralNetworksMemoryDesc_free_fn)(ANeuralNetworksMemoryDesc *desc);

typedef int (*ANeuralNetworksMemoryDesc_addInputRole_fn)(
  ANeuralNetworksMemoryDesc *desc, const ANeuralNetworksCompilation *compilation, int32_t index,
  float frequency);

typedef int (*ANeuralNetworksMemoryDesc_addOutputRole_fn)(
  ANeuralNetworksMemoryDesc *desc, const ANeuralNetworksCompilation *compilation, uint32_t index,
  float frequency);

typedef int (*ANeuralNetworksMemoryDesc_setDimensions_fn)(ANeuralNetworksMemoryDesc *desc,
                                                          uint32_t rank,
                                                          const uint32_t *dimensions);

typedef int (*ANeuralNetworksMemoryDesc_finish_fn)(ANeuralNetworksMemoryDesc *desc);

typedef int (*ANeuralNetworksMemory_createFromDesc_fn)(const ANeuralNetworksMemoryDesc *desc,
                                                       ANeuralNetworksMemory **memory);

typedef int (*ANeuralNetworksMemory_copy_fn)(const ANeuralNetworksMemory *src,
                                             const ANeuralNetworksMemory *dst);
#endif // __ANDROID_API__ >= 30
#endif // __NEURAL_NETWORKS_TYPES_H__
