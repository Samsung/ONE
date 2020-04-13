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
 * @file     CustomOps.h
 * @brief    This file contains registration of custom operands
 * @ingroup  COM_AI_RUNTIME
 */

#ifndef __NNFW_TFLITE_EXT_KERNELS_CUSTOM_OP_H__
#define __NNFW_TFLITE_EXT_KERNELS_CUSTOM_OP_H__

#include "tensorflow/lite/context.h"
#include "tflite/ext/kernels/SquaredDifference.h"

namespace nnfw
{
namespace tflite
{
namespace custom
{

#define REGISTER_FUNCTION(Name)             \
  TfLiteRegistration *Register_##Name(void) \
  {                                         \
    static TfLiteRegistration r = {};       \
    r.init = Name::Init##Name;              \
    r.free = Name::Free##Name;              \
    r.prepare = Name::Prepare##Name;        \
    r.invoke = Name::Eval##Name;            \
    r.custom_name = #Name;                  \
    return &r;                              \
  }

REGISTER_FUNCTION(SquaredDifference)

#undef REGISTER_FUNCTION

} // namespace custom
} // namespace tflite
} // namespace nnfw

#endif // __NNFW_TFLITE_EXT_KERNELS_CUSTOM_OP_H__
