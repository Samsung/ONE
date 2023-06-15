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

#include "OperationCloner.h"

#include <assert.h>

namespace onert
{
namespace ir
{

namespace
{

class OperationCloner : public OperationVisitor
{
public:
#define OP(Name) void visit(const operation::Name &o) override;
#include "ir/Operations.lst"
#undef OP

public:
  std::unique_ptr<Operation> releaseClone();

private:
  std::unique_ptr<Operation> _return_op;
};

#define OP(Name)                                        \
  void OperationCloner::visit(const operation::Name &o) \
  {                                                     \
    assert(!_return_op);                                \
    _return_op = std::make_unique<operation::Name>(o);  \
  }
#include "ir/Operations.lst"
#undef OP

std::unique_ptr<Operation> OperationCloner::releaseClone()
{
  assert(_return_op);
  return std::move(_return_op);
}

} // namespace

std::unique_ptr<Operation> clone(const IOperation &operation)
{
  OperationCloner cloner;
  operation.accept(cloner);
  return cloner.releaseClone();
}

} // namespace ir
} // namespace onert
