/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

#include <cassert>
#include <stdexcept>

namespace moco
{
namespace onnx
{

loco::DataType as_loco_datatype(const int32_t tensor_dtype)
{
  switch (tensor_dtype)
  {
    case 0: // UNDEFINED
      return loco::DataType::Unknown;
    case 1: // FLOAT
      return loco::DataType::FLOAT32;
    case 2: // UINT8
      return loco::DataType::U8;
    case 3: // INT8
      return loco::DataType::S8;
    case 4: // UINT16
      return loco::DataType::U16;
    case 5: // INT16
      return loco::DataType::S16;
    case 6: // INT32
      return loco::DataType::S32;
    case 7: // INT64
      return loco::DataType::S64;
    case 10: // FLOAT16
      return loco::DataType::FLOAT16;
    case 11: // DOUBLE
      return loco::DataType::FLOAT64;
    case 12: // UINT32
      return loco::DataType::U32;
    case 13: // UINT64
      return loco::DataType::U64;

    case 8:  // STRING
    case 9:  // BOOL
    case 14: // COMPLEX64
    case 15: // COMPLEX128
    case 16: // BFLOAT16
    default:
      break;
  }
  throw std::runtime_error{"Unsupported onnx dtype"};
}

std::string tensor_dtype_as_string(const int32_t tensor_dtype)
{
  switch (tensor_dtype)
  {
    case 0: // UNDEFINED
      return "UNDEFINED";
    case 1: // FLOAT
      return "FLOAT";
    case 2: // UINT8
      return "UINT8";
    case 3: // INT8
      return "INT8";
    case 4: // UINT16
      return "UINT16";
    case 5: // INT16
      return "INT16";
    case 6: // INT32
      return "INT32";
    case 7: // INT64
      return "INT64";
    case 8: // STRING
      return "STRING";
    case 9: // BOOL
      return "BOOL";
    case 10: // FLOAT16
      return "FLOAT16";
    case 11: // DOUBLE
      return "DOUBLE";
    case 12: // UINT32
      return "UINT32";
    case 13: // UINT64
      return "UINT64";
    case 14: // COMPLEX64
      return "COMPLEX64";
    case 15: // COMPLEX128
      return "COMPLEX128";
    case 16: // BFLOAT16
      return "BFLOAT16";
    default:
      break;
  }
  throw std::runtime_error{"Unsupported onnx dtype"};
}

} // namespace onnx
} // namespace moco
