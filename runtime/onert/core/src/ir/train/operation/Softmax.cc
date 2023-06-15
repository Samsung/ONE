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

#include "ir/train/operation/Softmax.h"

#include "ir/OperationVisitor.h"
#include "ir/train/TrainableOperationVisitor.h"

namespace onert
{
namespace ir
{
namespace train
{
namespace operation
{

std::unique_ptr<ITrainableOperation> Softmax::clone() const
{
  return std::make_unique<Softmax>(*this);
}

void Softmax::accept(OperationVisitor &v) const { v.visit(*this); }

void Softmax::accept(TrainableOperationVisitor &v) const { v.visit(*this); }

Softmax::Softmax(const OperationType &operation)
  : OperationType{operation.getInputs(), operation.getOutputs(), operation.param()}
{
  // DO NOTHING
}

} // namespace operation
} // namespace train
} // namespace ir
} // namespace onert
