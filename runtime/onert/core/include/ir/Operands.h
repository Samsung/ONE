/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_IR_OPERANDS_H__
#define __ONERT_IR_OPERANDS_H__

#include "ir/Index.h"
#include "ir/Operand.h"
#include "util/ObjectManager.h"

#include <memory>
#include <unordered_map>

namespace onert::ir
{

class Operands : public util::ObjectManager<OperandIndex, Operand>
{
public:
  Operands() = default;
  Operands(const Operands &obj);
  Operands(Operands &&) = default;
  Operands &operator=(const Operands &) = delete;
  Operands &operator=(Operands &&) = default;
  ~Operands() = default;
};

} // namespace onert::ir

#endif // __ONERT_MODEL_OPERAND_SET_H__
