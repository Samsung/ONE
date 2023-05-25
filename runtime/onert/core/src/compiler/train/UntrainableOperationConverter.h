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

#ifndef __ONERT_COMPILER_TRAIN_UNTRAINABLE_OPERATION_CONVERTER_H__
#define __ONERT_COMPILER_TRAIN_UNTRAINABLE_OPERATION_CONVERTER_H__

#include "ir/Operations.Include.h"
#include "ir/OperationVisitor.h"
#include "ir/train/TrainableGraph.h"

namespace onert
{
namespace compiler
{
namespace train
{

class UntrainableOperationConverter : public ir::OperationVisitor
{
public:
  UntrainableOperationConverter(ir::train::TrainableGraph &trainable_graph);

  std::unique_ptr<ir::train::ITrainableOperation> operator()(const ir::OperationIndex &index);

#define OP(InternalName) void visit(const ir::operation::InternalName &node);
#include "ir/Operations.lst"
#undef OP

protected:
  ir::train::TrainableGraph &_trainable_graph;
  std::unique_ptr<ir::train::ITrainableOperation> _return_op;
};

} // namespace train
} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_TRAIN_UNTRAINABLE_OPERATION_CONVERTER_H__
