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

#include "ir/train/operation/Permute.h"

#include "ir/OperationVisitor.h"
#include "ir/train/TrainableOperationVisitor.h"

#include <misc/polymorphic_downcast.h>

namespace onert
{
namespace ir
{
namespace train
{
namespace operation
{

std::unique_ptr<ITrainableOperation> Permute::clone() const
{
  return std::make_unique<Permute>(*this);
}

void Permute::accept(OperationVisitor &v) const { v.visit(*this); }

void Permute::accept(TrainableOperationVisitor &v) const { v.visit(*this); }

Permute::Permute(const OperationType &operation)
  : OperationType{operation.getInputs().at(0), operation.getOutputs().at(0),
                  operation.getPermuteType()}
{
  // DO NOTHING
}

} // namespace operation
} // namespace train
} // namespace ir
} // namespace onert
