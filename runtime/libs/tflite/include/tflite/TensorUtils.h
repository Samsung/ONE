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
 * @file    TensorUtils.h
 * @brief   This file contains utilities function
 * @ingroup COM_AI_RUNTIME
 */

#ifndef __NNFW_TFLITE_TENSOR_UTILS_H__
#define __NNFW_TFLITE_TENSOR_UTILS_H__

#include <tensorflow/lite/context.h>

namespace nnfw
{
namespace tflite
{

/**
 * @brief Get @c true if tensor type is kTfLiteFloat32, otherwise @c false
 * @param[in] tensor The tensor object to be compared
 * @return @c true if tensor type is kTfLiteFloat32, otherwise @c false
 */
inline bool isFloatTensor(const TfLiteTensor *tensor) { return tensor->type == kTfLiteFloat32; }

/**
 * @brief Get @c true if tensor is 4-D tensor and the first dimension length is 1,
 *        otherwise @c false
 * @param[in] tensor The tensor object to be compared
 * @return @c true if tensor is 4-D tensor and the first dimension length is 1, otherwise @c false
 */
inline bool isFeatureTensor(const TfLiteTensor *tensor)
{
  return (tensor->dims->size == 4) && (tensor->dims->data[0] == 1);
}

} // namespace tflite
} // namespace nnfw

#endif // __NNFW_TFLITE_TENSOR_UTILS_H__
