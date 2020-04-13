/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file     SquaredDifference.h
 * @brief    This file contains SquaredDifference namespace and SquaredDifference function
 *           definitions
 * @ingroup  COM_AI_RUNTIME
 */

#ifndef __NNFW_TFLITE_EXT_KERNELS_SQUARED_DIFFERENCE_H__
#define __NNFW_TFLITE_EXT_KERNELS_SQUARED_DIFFERENCE_H__

#include "tensorflow/lite/context.h"

namespace nnfw
{
namespace tflite
{
namespace custom
{
namespace SquaredDifference
{

/**
 * @brief Initialize SquaredDifference operand using the contents of buffer
 * @param[in] context The TfLite context
 * @param[in] buffer The buffer with contents
 * @param[in] length The buffer length
 * @return The void pointer for user data
 */
void *InitSquaredDifference(TfLiteContext *context, const char *buffer, size_t length);

/**
 * @brief Release any memory it might have allocated via 'InitSquaredDifference'
 * @param[in] context The TfLite context
 * @param[in] buffer The buffer with contents
 * @return N/A
 */
void FreeSquaredDifference(TfLiteContext *context, void *buffer);

/**
 * @brief Prepare the SquaredDifference operand for execution
 * @param[in] context The TfLite context
 * @param[in] node The operand node
 * @return The TfLite status
 */
TfLiteStatus PrepareSquaredDifference(TfLiteContext *context, TfLiteNode *node);

/**
 * @brief Evaluation the SquaredDifference operand for execution
 * @param[in] context The TfLite context
 * @param[in] node The operand node
 * @return The TfLite status
 */
TfLiteStatus EvalSquaredDifference(TfLiteContext *context, TfLiteNode *node);

} // namespace SquaredDifference
} // namespace custom
} // namespace tflite
} // namespace nnfw

#endif // __NNFW_TFLITE_EXT_KERNELS_SQUARED_DIFFERENCE_H__
