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

#include "UntrainableOperationConverter.h"

#include "ir/train/operation/UntrainableOperation.h"

#include <memory>

namespace onert
{
namespace compiler
{
namespace train
{

UntrainableOperationConverter::UntrainableOperationConverter(
  ir::train::TrainableGraph &trainable_graph)
  : _trainable_graph{trainable_graph}, _return_op{nullptr}
{
}

std::unique_ptr<ir::train::ITrainableOperation> UntrainableOperationConverter::
operator()(const ir::OperationIndex &index)
{
  const auto &op = _trainable_graph.operations().at(index);
  op.accept(*this);

  return std::move(_return_op);
}

#define OP(InternalName)                                                                         \
  void UntrainableOperationConverter::visit(const ir::operation::InternalName &node)             \
  {                                                                                              \
    _return_op =                                                                                 \
      std::make_unique<ir::train::operation::UntrainableOperation<ir::operation::InternalName>>( \
        node);                                                                                   \
  }
#include "ir/Operations.lst"
#undef OP

} // namespace train
} // namespace compiler
} // namespace onert
