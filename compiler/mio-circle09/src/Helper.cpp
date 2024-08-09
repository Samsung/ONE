/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "mio_circle/Helper.h"

#include <algorithm>
#include <sstream>

namespace mio
{
namespace circle
{

/**
 * This will provide v3/v3a/v3b format neutral BuiltinOperator
 * NOTE circle has minus value opcode (252~254 as uint8_t)
 *      we cannot use std::max() like tflite as deprecated_builtin_code can be
 *      minus and builtin_code being 0 for v0.3 files.
 */
::circle::BuiltinOperator builtin_code_neutral(const ::circle::OperatorCode *opcode)
{
  assert(opcode != nullptr);
  if (opcode->deprecated_builtin_code() == 127)
  {
    assert(opcode->builtin_code() >= 127);
    return opcode->builtin_code();
  }
  // There was no 255(-1) value in v0.3
  assert(opcode->deprecated_builtin_code() != -1);
  return static_cast<::circle::BuiltinOperator>(opcode->deprecated_builtin_code());
}

bool is_valid(const ::circle::OperatorCode *opcode)
{
  // Valid Range : BuiltinOperator_MIN <= deprecated_builtin_code <= 127
  const int8_t deprecated_builtin_code = opcode->deprecated_builtin_code();
  if (deprecated_builtin_code < ::circle::BuiltinOperator_MIN)
    return false;
  // There was no 255(-1) value in v0.3
  if (deprecated_builtin_code == -1)
    return false;

  const ::circle::BuiltinOperator builtin_code = opcode->builtin_code();
  if (!(::circle::BuiltinOperator_MIN <= builtin_code &&
        builtin_code <= ::circle::BuiltinOperator_MAX))
    return false;

  return true;
}

bool is_custom(const ::circle::OperatorCode *opcode)
{
  ::circle::BuiltinOperator code = builtin_code_neutral(opcode);
  return (code == ::circle::BuiltinOperator_CUSTOM);
}

std::string opcode_name(const ::circle::OperatorCode *opcode)
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

  ::circle::BuiltinOperator code = builtin_code_neutral(opcode);
  return ::circle::EnumNameBuiltinOperator(code);
}

const char *tensor_type(const ::circle::Tensor *tensor)
{
  return ::circle::EnumNameTensorType(tensor->type());
}

const char *tensor_name(const ::circle::Tensor *tensor)
{
  if (tensor->name() == nullptr || std::string(tensor->name()->c_str()).empty())
    return "(noname)";

  return tensor->name()->c_str();
}

} // namespace circle
} // namespace mio
