/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _TFLITE_NATIVEWRAPPER_H_
#define _TFLITE_NATIVEWRAPPER_H_

#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/mutable_op_resolver.h"

#ifdef __cplusplus
extern "C"
{
#endif /*__cplusplus*/

  typedef enum
  {
    /** 32-bit signed integer. */
    INT32 = 1,

    /** 32-bit single precision floating point. */
    FLOAT32 = 2,

    /** 8-bit unsigned integer. */
    UINT8 = 3,

    /** 64-bit signed integer. */
    INT64 = 4
  } TFLiteNativeType;

  void tflite_interpreter_setNumThreads(long *interpreterHandle, int numThreads);

  long long tflite_flatbuffermodel_BuildFromFile(char *modelPath);

  long long tflite_builder_interpreterBuilder(long *modelHandle);

  void *tflite_interpreter_run(long *interpreterHandle, void *values, int inputLength,
                               int dataType);

#ifdef __cplusplus
}
#endif /*__cplusplus*/

#endif /*_TFLITE_NATIVEWRAPPER_H_*/
