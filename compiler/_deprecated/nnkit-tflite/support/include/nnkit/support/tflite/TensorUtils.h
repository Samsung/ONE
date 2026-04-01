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

#ifndef __NNKIT_SUPPORT_TENSOR_UTILS_H__
#define __NNKIT_SUPPORT_TENSOR_UTILS_H__

#include <tensorflow/lite/context.h>
#include <nncc/core/ADT/tensor/Shape.h>

namespace nnkit
{
namespace support
{
namespace tflite
{

nncc::core::ADT::tensor::Shape tensor_shape(const TfLiteTensor *t);

} // namespace tflite
} // namespace support
} // namespace nnkit

#endif // __NNKIT_SUPPORT_TFLITE_TENSOR_UTILS_H__
