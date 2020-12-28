/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ir/Operands.h"

#include <memory>
#include "util/logging.h"

namespace onert
{
namespace ir
{

Operands::Operands(const Operands &obj)
{
  obj.iterate([&](const OperandIndex &index, const Operand &operand) {
    _objects.emplace(index, std::make_unique<Operand>(operand));
  });
  _next_index = obj._next_index;
}

} // namespace ir
} // namespace onert
