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
 * @file     Quantization.h
 * @brief    This file contains BitwiseIntToFloat union and quantization related
 * @ingroup  COM_AI_RUNTIME
 */

#ifndef __NNFW_TFLITE_QUANTIZATION_H__
#define __NNFW_TFLITE_QUANTIZATION_H__

/**
 * @brief Union to provide bitwise conversion of integer and float
 */
union BitwiseIntToFloat {
  int i;
  float f;
};

static const float FLOAT_NEAREST_TO_1 = BitwiseIntToFloat{0x3f7fffff}.f;

#include "tensorflow/lite/context.h"

/**
 * @brief   Get TfLiteQuantizationParams object with default values
 * @return  TfLiteQuantizationParams object
 */
TfLiteQuantizationParams make_default_quantization(void);

#endif // __NNFW_TFLITE_QUANTIZATION_H__
