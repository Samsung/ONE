/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "mir/Operation.h"
#include "mir/Visitor.h"
#include "mir/OpDefs.h"

#include <algorithm>

namespace mir
{

void Operation::Output::removeUse(Operation::Use use)
{
  auto it = std::remove(_uses.begin(), _uses.end(), use);
  _uses.erase(it);
}

void Operation::Output::replaceAllUsesWith(mir::Operation::Output *new_def)
{
  for (auto use : _uses)
  {
    use.getNode()->_inputs[use.getIndex()] = new_def;
    new_def->addUse(use);
  }
  _uses.clear();
}

Operation::Operation(Type type, const std::vector<Output *> &inputs, std::size_t num_outputs)
  : _type(type)
{
  for (std::size_t i = 0; i < inputs.size(); ++i)
  {
    inputs[i]->addUse(Use(this, i));
    _inputs.push_back(inputs[i]);
  }
  for (std::size_t i = 0; i < num_outputs; ++i)
  {
    _outputs.emplace_back(this, i);
  }
}

void Operation::accept(IVisitor *v)
{
  switch (getType())
  {
#define HANDLE_OP(OpType, OpClass)                 \
  case Type::OpType:                               \
    v->visit(dynamic_cast<ops::OpClass &>(*this)); \
    break;
#include "mir/Operations.inc"
#undef HANDLE_OP
    default:
      assert(false && "OP not defined!");
  }
}

const std::string &getTypeName(Operation::Type type)
{
  switch (type)
  {
#define HANDLE_OP(OpType, OpClass)          \
  case Operation::Type::OpType:             \
  {                                         \
    static const std::string name(#OpType); \
    return name;                            \
  }
#include "mir/Operations.inc"
#undef HANDLE_OP
  }
  throw std::runtime_error("unexpected opcode");
}

} // namespace mir
