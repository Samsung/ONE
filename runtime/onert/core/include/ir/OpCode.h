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

#ifndef __ONERT_IR_OP_CODE_H__
#define __ONERT_IR_OP_CODE_H__

#include <functional>
#include <stdint.h>
#include <string>

namespace onert::ir
{

enum class OpCode
{
  Invalid,             //< Unused
#define OP(Name) Name, //< All operations
#include "ir/Operations.lst"
#undef OP
  COUNT
};

const char *toString(OpCode opcode);
OpCode toOpCode(const std::string str);

} // namespace onert::ir

namespace std
{

template <> struct hash<onert::ir::OpCode>
{
  size_t operator()(onert::ir::OpCode value) const noexcept
  {
    using type = typename std::underlying_type<onert::ir::OpCode>::type;
    return hash<type>()(static_cast<type>(value));
  }
};

} // namespace std

#endif // __ONERT_IR_OP_CODE_H__
