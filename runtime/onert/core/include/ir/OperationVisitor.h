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

#ifndef __ONERT_IR_OPERATION_VISITOR_H__
#define __ONERT_IR_OPERATION_VISITOR_H__

#include "ir/Operations.Include.h"
#include "ir/OpSequence.h"

namespace onert
{
namespace ir
{

struct OperationVisitor
{
  virtual ~OperationVisitor() = default;

#define OP(InternalName) \
  virtual void visit(const operation::InternalName &) {}
#include "ir/Operations.lst"
#undef OP

  // This OpSequence node should be handled specially so that
  // Op.lst doesn't have OpSequence
  // TODO Remove by pushing it down to derived classes.
  virtual void visit(const OpSequence &)
  {
    throw std::runtime_error{
      "OperationVisitor: This does not privide visit function in OpSequence"};
  }
};

} // namespace ir
} // namespace onert

#endif // __ONERT_IR_OPERATION_VISITOR_H__
