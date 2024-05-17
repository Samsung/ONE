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

#ifndef __ONERT_IR_TRAIN_ITRAINABLE_OPERATION_H__
#define __ONERT_IR_TRAIN_ITRAINABLE_OPERATION_H__

#include "ir/IOperation.h"

namespace onert
{
namespace ir
{
namespace train
{

struct TrainableOperationVisitor;

// NOTE Virtual inheritance is introduced because trainable operations inherit
//      `ITrainableOperation` and `Operation` which inherit `IOperation`.
class ITrainableOperation : virtual public IOperation
{
public:
  virtual ~ITrainableOperation() = default;

public:
  virtual std::unique_ptr<ITrainableOperation> clone() const = 0;
  virtual void accept(OperationVisitor &v) const override = 0;
  virtual void accept(TrainableOperationVisitor &v) const = 0;
  virtual bool hasTrainableParameter() const = 0;

  // Update of the node will be disabled during traning
  virtual void disableWeightsUpdate() = 0;
  // Note that the update is disabled by default
  // Update of the node will be enabled during traning
  virtual void enableWeightsUpdate() = 0;
  // Check if the node is trainable
  virtual bool isWeightsUpdateEnabled() const = 0;

  // Mark the node as needed for backward propagation part of the traning
  virtual void enableBackward() = 0;
  // Check if the nodes is required for backward propagation part of the traning.
  // If returned false, it means that there are no node before (in topological sense) which is
  // trainable.
  virtual bool isRequiredForBackward() const = 0;
};

} // namespace train
} // namespace ir
} // namespace onert

#endif // __ONERT_IR_TRAIN_ITRAINABLE_OPERATION_H__
