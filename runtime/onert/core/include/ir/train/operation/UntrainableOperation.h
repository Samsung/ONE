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

#ifndef __ONERT_IR_TRAIN_OPERATION_UNTRAINABLE_OPERATION_H__
#define __ONERT_IR_TRAIN_OPERATION_UNTRAINABLE_OPERATION_H__

#include "ir/train/ITrainableOperation.h"

#include "ir/OperationVisitor.h"
#include "ir/train/TrainableOperationVisitor.h"

#include <type_traits>

namespace onert
{
namespace ir
{
namespace train
{
namespace operation
{

// `UntrainableOperation` wraps operations that are not yet supported for training.
// This class can be removed if all operations are supported for training.
template <typename OperationType,
          typename = std::enable_if_t<std::is_base_of<Operation, OperationType>::value>>
class UntrainableOperation : public OperationType, public ITrainableOperation
{
public:
  UntrainableOperation(const OperationType &operation) : OperationType{operation} {}
  virtual ~UntrainableOperation() = default;

public:
  std::unique_ptr<ITrainableOperation> clone() const override
  {
    return std::make_unique<UntrainableOperation<OperationType>>(*this);
  }
  void accept(OperationVisitor &v) const override { v.visit(*this); }
  void accept(TrainableOperationVisitor &) const override
  {
    throw std::runtime_error(OperationType::name() + "operation is not trainable yet");
  }
  bool hasTrainableParameter() const override { return false; }
};

} // namespace operation
} // namespace train
} // namespace ir
} // namespace onert

#endif // __ONERT_IR_TRAIN_OPERATION_UNTRAINABLE_OPERATION_H__
