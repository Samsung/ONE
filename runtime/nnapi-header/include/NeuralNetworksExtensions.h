/*
 * Copyright (C) 2019 The Android Open Source Project
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

#ifndef ANDROID_FRAMEWORKS_ML_NN_RUNTIME_NEURAL_NETWORKS_EXTENSIONS_H
#define ANDROID_FRAMEWORKS_ML_NN_RUNTIME_NEURAL_NETWORKS_EXTENSIONS_H

#include "NeuralNetworks.h"

/******************************************************************
 *
 * IMPORTANT NOTICE:
 *
 *   This file is not intended for use by general developers -- only
 *   by OEM applications.
 *
 *   Extensions source AND binary code relies on the definitions
 *   here to be FROZEN ON ALL UPCOMING PLATFORM RELEASES.
 *
 *   - DO NOT MODIFY ENUMS (EXCEPT IF YOU ADD NEW 32-BIT VALUES)
 *   - DO NOT MODIFY CONSTANTS OR FUNCTIONAL MACROS
 *   - DO NOT CHANGE THE SIGNATURE OF FUNCTIONS IN ANY WAY
 *   - DO NOT CHANGE THE LAYOUT OR SIZE OF STRUCTURES
 */

__BEGIN_DECLS

#if __ANDROID_API__ >= 29

/**
 * Queries whether an extension is supported by the driver implementation of the specified device.
 *
 * @param device The representation of the specified device.
 * @param extension The extension name.
 * @param isExtensionSupported The boolean value indicating whether the extension is supported.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *
 * Available since API level 29.
 */
int ANeuralNetworksDevice_getExtensionSupport(const ANeuralNetworksDevice* device,
                                              const char* extensionName, bool* isExtensionSupported)
        __INTRODUCED_IN(29);

/**
 * Creates an operand type from an extension name and an extension operand code.
 *
 * See {@link ANeuralNetworksModel} for information on multithreaded usage.
 *
 * Available since API level 29.
 *
 * @param model The model to contain the operand.
 * @param extensionName The extension name.
 * @param operandCodeWithinExtension The extension operand code.
 * @param type The operand type.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
int ANeuralNetworksModel_getExtensionOperandType(ANeuralNetworksModel* model,
                                                 const char* extensionName,
                                                 uint16_t operandCodeWithinExtension, int32_t* type)
        __INTRODUCED_IN(29);

/**
 * Creates an operation type from an extension name and an extension operation code.
 *
 * See {@link ANeuralNetworksModel} for information on multithreaded usage.
 *
 * Available since API level 29.
 *
 * @param model The model to contain the operation.
 * @param extensionName The extension name.
 * @param operationCodeWithinExtension The extension operation code.
 * @param type The operation type.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
int ANeuralNetworksModel_getExtensionOperationType(ANeuralNetworksModel* model,
                                                   const char* extensionName,
                                                   uint16_t operationCodeWithinExtension,
                                                   ANeuralNetworksOperationType* type)
        __INTRODUCED_IN(29);

/**
 * Sets extension operand parameters.
 *
 * Available since API level 29.
 *
 * @param model The model to be modified.
 * @param index The index of the model operand we're setting.
 * @param data A pointer to the extension operand data.
 *             The data does not have to outlive the call to this function.
 * @param length The size in bytes of the data value.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
int ANeuralNetworksModel_setOperandExtensionData(ANeuralNetworksModel* model, int32_t index,
                                                 const void* data, size_t length)
        __INTRODUCED_IN(29);

#endif  // __ANDROID_API__ >= 29

__END_DECLS

#endif  // ANDROID_FRAMEWORKS_ML_NN_RUNTIME_NEURAL_NETWORKS_EXTENSIONS_H
