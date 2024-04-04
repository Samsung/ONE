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

#include <stdexcept>

circle::Padding as_circle_padding(const circlechef::Padding &value)
{
  switch (value)
  {
    case circlechef::SAME:
      return circle::Padding_SAME;
    case circlechef::VALID:
      return circle::Padding_VALID;
    default:
      break;
  }

  throw std::runtime_error{"Unknown padding value"};
}

circle::ActivationFunctionType as_circle_activation(const circlechef::Activation &value)
{
  switch (value)
  {
    case circlechef::NONE:
      return circle::ActivationFunctionType_NONE;
    case circlechef::RELU:
      return circle::ActivationFunctionType_RELU;
    case circlechef::RELU6:
      return circle::ActivationFunctionType_RELU6;
    default:
      break;
  }

  throw std::runtime_error{"Unknown activation"};
}

circle::TensorType as_circle_tensortype(const circlechef::TensorType &value)
{
  switch (value)
  {
    case circlechef::FLOAT32:
      return circle::TensorType_FLOAT32;
    case circlechef::INT64:
      return circle::TensorType_INT64;
    case circlechef::INT32:
      return circle::TensorType_INT32;
    case circlechef::INT16:
      return circle::TensorType_INT16;
    case circlechef::INT4:
      return circle::TensorType_INT4;
    case circlechef::UINT8:
      return circle::TensorType_UINT8;
    case circlechef::UINT4:
      return circle::TensorType_UINT4;
    case circlechef::STRING:
      return circle::TensorType_STRING;
    case circlechef::BOOL:
      return circle::TensorType_BOOL;
    default:
      break;
  }

  throw std::runtime_error{"Unknown tensor type"};
}
