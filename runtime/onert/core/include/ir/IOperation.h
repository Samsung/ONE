/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_IR_IOPERATION_H__
#define __ONERT_IR_IOPERATION_H__

#include <memory>

#include "ir/Index.h"
#include "ir/OpCode.h"
#include "ir/OperandIndexSequence.h"

namespace onert
{
namespace ir
{

struct OperationVisitor;

struct IOperation
{
  virtual ~IOperation() = default;

  virtual void accept(OperationVisitor &v) const = 0;
  virtual std::string name() const { return std::string{toString(opcode())}; }
  virtual OpCode opcode() const = 0;
  virtual std::unique_ptr<IOperation> clone() const
  {
    throw std::runtime_error{name() + "::clone() not supported"};
    // return std::unique_ptr<IOperation>();
  }

  virtual void replaceInputs(const OperandIndex &from, const OperandIndex &to) = 0;
  virtual void replaceOutputs(const OperandIndex &from, const OperandIndex &to) = 0;
  virtual const OperandIndexSequence &getInputs() const = 0;
  virtual const OperandIndexSequence &getOutputs() const = 0;
};

} // namespace ir
} // namespace onert

#endif // __ONERT_IR_IOPERATION_H__
