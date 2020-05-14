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

#ifndef __ONERT_IR_OPERATION_REVERSE_H__
#define __ONERT_IR_OPERATION_REVERSE_H__

#include "ir/Operation.h"

namespace onert
{
namespace ir
{
namespace operation
{

class Reverse : public Operation
{
public:
  enum Input
  {
    INPUT = 0,
    AXIS = 1
  };

public:
  Reverse(const OperandIndexSequence &inputs, const OperandIndexSequence &outputs);

public:
  void accept(OperationVisitor &v) const override;
  OpCode opcode() const final { return OpCode::Reverse; }
};

} // namespace operation
} // namespace ir
} // namespace onert

#endif // __ONERT_IR_OPERATION_REVERSE_H__
