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
 * @file     Assert.h
 * @brief    This file contains helper function of assertion
 * @ingroup  COM_AI_RUNTIME
 */

#ifndef __NNFW_TFLITE_ASSERT_H__
#define __NNFW_TFLITE_ASSERT_H__

#include "tensorflow/lite/context.h"

#include <sstream>

#define STR_DETAIL(value) #value
#define STR(value) STR_DETAIL(value)

#define TFLITE_ENSURE(exp)                                             \
  {                                                                    \
    const TfLiteStatus status = (exp);                                 \
                                                                       \
    if (status != kTfLiteOk)                                           \
    {                                                                  \
      std::ostringstream ss;                                           \
      ss << #exp << " failed (" << __FILE__ << ":" << __LINE__ << ")"; \
      throw std::runtime_error{ss.str()};                              \
    }                                                                  \
  }

#endif // __NNFW_TFLITE_ASSERT_H__
