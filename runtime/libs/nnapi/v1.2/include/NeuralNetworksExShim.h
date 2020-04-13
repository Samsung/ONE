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
 * @file     NeuralNetworksExShim.h
 * @brief    This file contains an actual implementation of
 *           ANeuralNetworksModel_addOperationEx function
 */

#ifndef __NEURAL_NETWORKS_EX_SHIM_H__
#define __NEURAL_NETWORKS_EX_SHIM_H__

#include "NeuralNetworks.h"
#include "NeuralNetworksEx.h"
#include "NeuralNetworksLoadHelpers.h"

typedef int (*ANeuralNetworksModel_addOperationEx_fn)(ANeuralNetworksModel *model,
                                                      ANeuralNetworksOperationTypeEx type,
                                                      uint32_t inputCount, const uint32_t *inputs,
                                                      uint32_t outputCount,
                                                      const uint32_t *outputs);

/**
 * @brief Add an extended operation to a model.
 *
 * @param[in] model The model to be modified.
 * @param[in] type The type of extended operation.
 * @param[in] inputCount The number of entries in the inputs array.
 * @param[in] inputs An array of indexes identifying each operand.
 * @param[in] outputCount The number of entries in the outputs array.
 * @param[in] outputs An array of indexes identifying each operand.
 *
 * @note The operands specified by inputs and outputs must have been
 *       previously added by calls to {@link ANeuralNetworksModel_addOperand}.\n
 *       Attempting to modify a model once {@link ANeuralNetworksModel_finish}
 *       has been called will return an error.\n
 *       See {@link ANeuralNetworksModel} for information on multithreaded usage.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */

inline int ANeuralNetworksModel_addOperationEx(ANeuralNetworksModel *model,
                                               ANeuralNetworksOperationTypeEx type,
                                               uint32_t inputCount, const uint32_t *inputs,
                                               uint32_t outputCount, const uint32_t *outputs)
{
  LOAD_FUNCTION(ANeuralNetworksModel_addOperationEx);
  EXECUTE_FUNCTION_RETURN(model, type, inputCount, inputs, outputCount, outputs);
}

#endif // __NEURAL_NETWORKS_EX_SHIM_H__
