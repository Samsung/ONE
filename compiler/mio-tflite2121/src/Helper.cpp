/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

#include "mio_tflite2121/Helper.h"

#include <sstream>

namespace mio
{
namespace tflite
{

/**
 * This will provide v3/v3a format neutral BuiltinOperator
 *
 * This function referenced
 * https://github.com/tensorflow/tensorflow/blob/7d12007d7800d3714a02e05059f3ea602d1aec78/tensorflow/lite/schema/schema_utils.cc
 */
::tflite::BuiltinOperator builtin_code_neutral(const ::tflite::OperatorCode *opcode)
{
  assert(opcode != nullptr);
  return std::max(opcode->builtin_code(),
                  static_cast<::tflite::BuiltinOperator>(opcode->deprecated_builtin_code()));
}

bool is_valid(const ::tflite::OperatorCode *opcode)
{
  // Valid Range : 0 <= deprecated_builtin_code <= 127
  const int8_t deprecated_builtin_code = opcode->deprecated_builtin_code();
  if (deprecated_builtin_code < 0)
    return false;

  const ::tflite::BuiltinOperator builtin_code = opcode->builtin_code();
  if (!(::tflite::BuiltinOperator_MIN <= builtin_code &&
        builtin_code <= ::tflite::BuiltinOperator_MAX))
    return false;

  return true;
}

bool is_custom(const ::tflite::OperatorCode *opcode)
{
  ::tflite::BuiltinOperator code = builtin_code_neutral(opcode);
  return (code == ::tflite::BuiltinOperator_CUSTOM);
}

std::string opcode_name(const ::tflite::OperatorCode *opcode)
{
  assert(opcode);

  if (!is_valid(opcode))
  {
    std::ostringstream oss;
    oss << "(invalid)";
    return oss.str();
  }

  if (is_custom(opcode))
  {
    if (!opcode->custom_code())
      return "(invalid custom)";

    std::string custom_op = "CUSTOM(";
    custom_op += opcode->custom_code()->c_str();
    custom_op += ")";
    return custom_op;
  }

  ::tflite::BuiltinOperator code = builtin_code_neutral(opcode);
  return ::tflite::EnumNameBuiltinOperator(code);
}

const char *tensor_type(const ::tflite::Tensor *tensor)
{
  return ::tflite::EnumNameTensorType(tensor->type());
}

const char *tensor_name(const ::tflite::Tensor *tensor)
{
  static const char *kEmptyTensorName = "(noname)";

  auto name = tensor->name();
  if (name)
    return name->c_str();

  return kEmptyTensorName;
}

} // namespace tflite
} // namespace mio
