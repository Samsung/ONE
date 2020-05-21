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

namespace tflchef
{

tflchef::TensorType as_tflchef_type(const tflite::TensorType type)
{
  switch (type)
  {
    case tflite::TensorType_FLOAT32:
      return tflchef::FLOAT32;
    case tflite::TensorType_INT32:
      return tflchef::INT32;
    case tflite::TensorType_INT64:
      return tflchef::INT64;
    case tflite::TensorType_UINT8:
      return tflchef::UINT8;
    case tflite::TensorType_BOOL:
      return tflchef::BOOL;
    // TODO handle other types
    // TensorType_FLOAT16
    // TensorType_STRING
    // TensorType_INT16
    // TensorType_COMPLEX64
    default:
      throw std::runtime_error{"unsupported tensor type"};
  }
}

tflchef::Activation as_tflchef_activation(const tflite::ActivationFunctionType type)
{
  switch (type)
  {
    case tflite::ActivationFunctionType_NONE:
      return tflchef::NONE;
    case tflite::ActivationFunctionType_RELU:
      return tflchef::RELU;
    case tflite::ActivationFunctionType_RELU_N1_TO_1:
      return tflchef::RELU_N1_TO_1;
    case tflite::ActivationFunctionType_RELU6:
      return tflchef::RELU6;
    // TODO handle other types
    // ActivationFunctionType_TANH
    // ActivationFunctionType_SIGN_BIT
    default:
      throw std::runtime_error{"unsupported activation type"};
  }
}

tflchef::Padding as_tflchef_padding(const tflite::Padding padding)
{
  switch (padding)
  {
    case tflite::Padding_SAME:
      return tflchef::SAME;
    case tflite::Padding_VALID:
      return tflchef::VALID;
    default:
      throw std::runtime_error{"unsupported padding"};
  }
}

tflchef::MirrorPadMode as_tflchef_mirrorpadmode(const tflite::MirrorPadMode mode)
{
  switch (mode)
  {
    case tflite::MirrorPadMode_REFLECT:
      return tflchef::REFLECT;
    case tflite::MirrorPadMode_SYMMETRIC:
      return tflchef::SYMMETRIC;
    default:
      throw std::runtime_error{"Unknown mirrorpad mode"};
  }
}

} // namespace tflchef
