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

#ifndef __ONERT_IR_ITRAINABLE_OPERATION_H__
#define __ONERT_IR_ITRAINABLE_OPERATION_H__

#include "ir/Operation.h"

namespace onert
{
namespace ir
{

struct OperationVisitor;

namespace train
{

struct TrainableOperationVisitor;

class ITrainableOperation : public IOperation
{
public:
  virtual ~ITrainableOperation() = default;

public:
  virtual void accept(TrainableOperationVisitor &v) const = 0;
  // TODO Add virtual methods related to training

public:
  void replaceInputs(const OperandIndex &from, const OperandIndex &to) override
  {
    operation().replaceInputs(from, to);
  }
  void replaceOutputs(const OperandIndex &from, const OperandIndex &to) override
  {
    operation().replaceOutputs(from, to);
  }
  const OperandIndexSequence &getInputs() const override { return operation().getInputs(); }
  const OperandIndexSequence &getOutputs() const override { return operation().getOutputs(); }

public:
  virtual Operation &operation() const = 0;
};

} // namespace train
} // namespace ir
} // namespace onert

#endif // __ONERT_IR_ITRAINABLE_OPERATION_H__
