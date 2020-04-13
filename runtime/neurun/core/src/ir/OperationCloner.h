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

#ifndef __NEURUN_IR_OPERATION_CLONER_H__
#define __NEURUN_IR_OPERATION_CLONER_H__

#include "memory"
#include "ir/OperationVisitor.h"
#include "ir/Operation.h"

namespace neurun
{
namespace ir
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

} // namespace ir
} // namespace neurun

#endif // __NEURUN_IR_OPERATION_CLONER_H__
