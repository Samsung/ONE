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

#ifndef __ACTIVATION_UTILS_H__
#define __ACTIVATION_UTILS_H__

#include "Logging.h"

#define ANDROID_NN_MACRO_DISPATCH_INTERNAL(macro) \
  case (int32_t)FusedActivationFunc::NONE:        \
    macro(kNone);                                 \
    break;                                        \
  case (int32_t)FusedActivationFunc::RELU:        \
    macro(kRelu);                                 \
    break;                                        \
  case (int32_t)FusedActivationFunc::RELU1:       \
    macro(kRelu1);                                \
    break;                                        \
  case (int32_t)FusedActivationFunc::RELU6:       \
    macro(kRelu6);                                \
    break;

#define ANDROID_NN_MACRO_DISPATCH(macro)                          \
  switch (activation)                                             \
  {                                                               \
    ANDROID_NN_MACRO_DISPATCH_INTERNAL(macro)                     \
    default:                                                      \
      LOG(ERROR) << "Unsupported fused activation function type"; \
      return false;                                               \
  }

#define ANDROID_NN_MACRO_DISPATCH_WITH_DELETE(macro)              \
  switch (activation)                                             \
  {                                                               \
    ANDROID_NN_MACRO_DISPATCH_INTERNAL(macro)                     \
    default:                                                      \
      LOG(ERROR) << "Unsupported fused activation function type"; \
      if (im2colByteSize > kStaticBufferSize)                     \
      {                                                           \
        delete[] im2colData;                                      \
      }                                                           \
      return false;                                               \
  }

#endif // __ACTIVATION_UTILS_H__
