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

#ifndef __ONERT_IR_TRAIN_OPERATION_ELEMENTWISE_ACTIVATION_H__
#define __ONERT_IR_TRAIN_OPERATION_ELEMENTWISE_ACTIVATION_H__

#include "ir/operation/ElementwiseActivation.h"
#include "ir/train/ITrainableOperation.h"

namespace onert
{
namespace ir
{
namespace train
{
namespace operation
{

class ElementwiseActivation : public ITrainableOperation
{
private:
  using OperationType = ir::operation::ElementwiseActivation;

public:
  ElementwiseActivation(OperationType &operation, const OperandIndexSequence &training_inputs);

public:
  std::unique_ptr<ITrainableOperation> clone(Operation &) const override;
  void accept(OperationVisitor &v) const override;
  void accept(TrainableOperationVisitor &v) const override;
  virtual OpCode opcode() const final { return _operation.opcode(); }

public:
  const OperationType::Param &param() const { return _operation.param(); }

public:
  const Operation &operation() const final { return _operation; }

private:
  Operation &operation() final { return _operation; }

private:
  OperationType &_operation;
  OperandIndexSequence _training_inputs;
};

} // namespace operation
} // namespace train
} // namespace ir
} // namespace onert

#endif // __ONERT_IR_TRAIN_OPERATION_ELEMENTWISE_ACTIVATION_H__
