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

#ifndef __ONERT_IR_OPERATION_LOSS_H__
#define __ONERT_IR_OPERATION_LOSS_H__

#include "ir/Operation.h"
#include "ir/train/LossCode.h"
#include "ir/train/LossInfo.h"

namespace onert
{
namespace ir
{
namespace operation
{

class Loss : public Operation
{
public:
  enum Input
  {
    Y_PRED = 0,
    Y_TRUE = 1
    // TODO Add more inputs if necessary
  };

public:
  Loss(const OperandIndexSequence &inputs, const OperandIndexSequence &outputs,
       const train::LossInfo &info);

public:
  void accept(OperationVisitor &v) const override;
  OpCode opcode() const final { return OpCode::Loss; }
  std::string name() const override { return toString(_param.loss_code) + toString(opcode()); };

public:
  const train::LossInfo &param() const { return _param; }

private:
  train::LossInfo _param;
};

} // namespace operation
} // namespace ir
} // namespace onert

#endif // __ONERT_IR_OPERATION_LOSS_H__
