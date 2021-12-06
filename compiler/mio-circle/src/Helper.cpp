/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <sstream>

namespace mio
{
namespace circle
{

bool is_valid(const ::circle::OperatorCode *opcode)
{
  ::circle::BuiltinOperator code = opcode->builtin_code();
  return (::circle::BuiltinOperator_MIN <= code && code <= ::circle::BuiltinOperator_MAX);
}

bool is_custom(const ::circle::OperatorCode *opcode)
{
  ::circle::BuiltinOperator code = opcode->builtin_code();
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

  ::circle::BuiltinOperator code = opcode->builtin_code();
  return ::circle::EnumNameBuiltinOperator(code);
}

const char *tensor_type(const ::circle::Tensor *tensor)
{
  return ::circle::EnumNameTensorType(tensor->type());
}

const char *tensor_name(const ::circle::Tensor *tensor)
{
  static const char *kEmptyTensorName = "(noname)";

  auto name = tensor->name();
  if (name)
    return name->c_str();

  return kEmptyTensorName;
}

} // namespace circle
} // namespace mio
