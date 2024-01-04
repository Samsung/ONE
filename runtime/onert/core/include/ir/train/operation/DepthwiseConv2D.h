/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_IR_TRAIN_OPERATION_DEPTHWISE_CONV2D_H__
#define __ONERT_IR_TRAIN_OPERATION_DEPTHWISE_CONV2D_H__

#include "ir/operation/DepthwiseConv2D.h"
#include "ir/train/ITrainableOperation.h"

namespace onert
{
namespace ir
{
namespace train
{
namespace operation
{

class DepthwiseConv2D : public ir::operation::DepthwiseConv2D, public ITrainableOperation
{
private:
  using OperationType = ir::operation::DepthwiseConv2D;

public:
  DepthwiseConv2D(const OperationType &operation);

public:
  std::unique_ptr<ITrainableOperation> clone() const override;
  void accept(OperationVisitor &v) const override;
  void accept(TrainableOperationVisitor &v) const override;
};

} // namespace operation
} // namespace train
} // namespace ir
} // namespace onert

#endif // __ONERT_IR_TRAIN_OPERATION_DEPTHWISE_CONV2D_H__
