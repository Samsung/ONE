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

#include "TrainableOperationConverter.h"

#include "ir/train/Operations.Include.h"
#include "util/Utils.h"

namespace onert
{
namespace compiler
{
namespace train
{

TrainableOperationConverter::TrainableOperationConverter(
  ir::train::TrainableGraph &tgraph, const ir::train::TrainingInfo *training_info)
  : UntrainableOperationConverter{tgraph}, _training_info{training_info}
{
  // Avoid unused-private-field error
  UNUSED_RELEASE(_training_info);
}

void TrainableOperationConverter::visit(const ir::operation::Conv2D &node)
{
  _return_op = std::make_unique<ir::train::operation::Conv2D>(node);
}

void TrainableOperationConverter::visit(const ir::operation::DepthwiseConv2D &node)
{
  _return_op = std::make_unique<ir::train::operation::DepthwiseConv2D>(node);
}

void TrainableOperationConverter::visit(const ir::operation::ElementwiseActivation &node)
{
  if (node.param().op_type == ir::operation::ElementwiseActivation::Type::RELU)
  {
    _return_op = std::make_unique<ir::train::operation::ElementwiseActivation>(node);
  }
  else
  {
    UntrainableOperationConverter::visit(node);
  }
}

void TrainableOperationConverter::visit(const ir::operation::FullyConnected &node)
{
  _return_op = std::make_unique<ir::train::operation::FullyConnected>(node);
}

void TrainableOperationConverter::visit(const ir::operation::Loss &node)
{
  _return_op = std::make_unique<ir::train::operation::Loss>(node, _training_info->lossInfo());
}

void TrainableOperationConverter::visit(const ir::operation::Permute &node)
{
  _return_op = std::make_unique<ir::train::operation::Permute>(node);
}

void TrainableOperationConverter::visit(const ir::operation::Pool2D &node)
{
  _return_op = std::make_unique<ir::train::operation::Pool2D>(node);
}

void TrainableOperationConverter::visit(const ir::operation::Reduce &node)
{
  _return_op = std::make_unique<ir::train::operation::Reduce>(node);
}

void TrainableOperationConverter::visit(const ir::operation::Reshape &node)
{
  _return_op = std::make_unique<ir::train::operation::Reshape>(node);
}

void TrainableOperationConverter::visit(const ir::operation::Softmax &node)
{
  _return_op = std::make_unique<ir::train::operation::Softmax>(node);
}

} // namespace train
} // namespace compiler
} // namespace onert
