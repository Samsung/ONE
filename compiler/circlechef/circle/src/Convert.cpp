/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

namespace circlechef
{

circlechef::TensorType as_circlechef_type(const circle::TensorType type)
{
  switch (type)
  {
    case circle::TensorType_FLOAT32:
      return circlechef::FLOAT32;
    case circle::TensorType_INT32:
      return circlechef::INT32;
    case circle::TensorType_INT64:
      return circlechef::INT64;
    case circle::TensorType_UINT8:
      return circlechef::UINT8;
    case circle::TensorType_BOOL:
      return circlechef::BOOL;
    case circle::TensorType_INT16:
      return circlechef::INT16;
    // TODO handle other types
    // TensorType_FLOAT16
    // TensorType_STRING
    // TensorType_COMPLEX64
    default:
      throw std::runtime_error{"unsupported tensor type"};
  }
}

circlechef::Activation as_circlechef_activation(const circle::ActivationFunctionType type)
{
  switch (type)
  {
    case circle::ActivationFunctionType_NONE:
      return circlechef::NONE;
    case circle::ActivationFunctionType_RELU:
      return circlechef::RELU;
    case circle::ActivationFunctionType_RELU6:
      return circlechef::RELU6;
    case circle::ActivationFunctionType_TANH:
      return circlechef::TANH;
    // TODO handle other types
    // ActivationFunctionType_RELU_N1_TO_1
    // ActivationFunctionType_SIGN_BIT
    default:
      throw std::runtime_error{"unsupported activation type"};
  }
}

circlechef::Padding as_circlechef_padding(const circle::Padding padding)
{
  switch (padding)
  {
    case circle::Padding_SAME:
      return circlechef::SAME;
    case circle::Padding_VALID:
      return circlechef::VALID;
    default:
      throw std::runtime_error{"unsupported padding"};
  }
}

} // namespace circlechef
