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

#include "Convert.h"

#include <stdexcept>

tflite::Padding as_tflite_padding(const tflchef::Padding &value)
{
  switch (value)
  {
    case tflchef::SAME:
      return tflite::Padding_SAME;
    case tflchef::VALID:
      return tflite::Padding_VALID;
    default:
      break;
  }

  throw std::runtime_error{"Unknown padding value"};
}

tflite::ActivationFunctionType as_tflite_activation(const tflchef::Activation &value)
{
  switch (value)
  {
    case tflchef::NONE:
      return tflite::ActivationFunctionType_NONE;
    case tflchef::RELU:
      return tflite::ActivationFunctionType_RELU;
    case tflchef::RELU_N1_TO_1:
      return tflite::ActivationFunctionType_RELU_N1_TO_1;
    case tflchef::RELU6:
      return tflite::ActivationFunctionType_RELU6;
    default:
      break;
  }

  throw std::runtime_error{"Unknown activation"};
}

tflite::TensorType as_tflite_tensortype(const tflchef::TensorType &value)
{
  switch (value)
  {
    case tflchef::FLOAT32:
      return tflite::TensorType_FLOAT32;
    case tflchef::INT32:
      return tflite::TensorType_INT32;
    case tflchef::UINT8:
      return tflite::TensorType_UINT8;
    case tflchef::INT64:
      return tflite::TensorType_INT64;
    case tflchef::BOOL:
      return tflite::TensorType_BOOL;
    default:
      break;
  }

  throw std::runtime_error{"Unknown tensor type"};
}
