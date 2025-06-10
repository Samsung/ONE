/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ir/OpCode.h"

#include <unordered_map>

namespace onert::ir
{

const char *toString(OpCode opcode)
{
  static const std::unordered_map<OpCode, const char *> map{{OpCode::Invalid, "Invalid"},
#define OP(Name) {OpCode::Name, #Name},
#include "ir/Operations.lst"
#undef OP
                                                            {OpCode::COUNT, "COUNT"}};
  return map.at(opcode);
}

OpCode toOpCode(const std::string str)
{
  // TODO: Apply Heterogeneous lookup for unordered containers (transparent hashing) since C++20
  //       to use `std::string_view` with lookup functions in unordered containers
  static const std::unordered_map<std::string, OpCode> map{
#define OP(Name) {#Name, OpCode::Name},
#include "ir/Operations.lst"
#undef OP
  };
  return map.at(str);
}

} // namespace onert::ir
