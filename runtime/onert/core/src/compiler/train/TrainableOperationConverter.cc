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

#include "ir/Operations.Include.h"
#include "ir/train/operation/ElementwiseActivation.h"
#include "ir/train/operation/Loss.h"
#include "util/Utils.h"

#include <memory>

namespace onert
{
namespace compiler
{
namespace train
{

TrainableOperationConverter::TrainableOperationConverter(
  ir::train::TrainableGraph &trainable_graph, const ir::train::TrainingInfo *training_info)
  : UntrainableOperationConverter{trainable_graph}, _training_info{training_info}
{
  assert(_training_info);
}

std::unique_ptr<ir::train::ITrainableOperation> TrainableOperationConverter::
operator()(const ir::OperationIndex &index)
{
  const auto &op = _trainable_graph.operations().at(index);
  op.accept(*this);

  return std::move(_return_op);
}

void TrainableOperationConverter::visit(const ir::operation::ElementwiseActivation &node)
{
  if (node.param().op_type == ir::operation::ElementwiseActivation::Type::RELU)
  {
    const auto &output_ind = node.getOutputs().at(0);
    const auto &output_obj = _trainable_graph.operands().at(output_ind);
    const auto &flex_shape = output_obj.shape();
    const auto &flex_type = output_obj.typeInfo();

    auto flex_ind = _trainable_graph.addOperand(flex_shape, flex_type);
    ir::OperandIndexSequence training_inputs{flex_ind};
    // TODO Remove const_cast
    _return_op = std::make_unique<ir::train::operation::ElementwiseActivation>(
      const_cast<ir::operation::ElementwiseActivation &>(node), training_inputs);
  }
  else
  {
    UntrainableOperationConverter::visit(node);
  }
}

void TrainableOperationConverter::visit(const ir::operation::Loss &)
{
  // TODO Remove this because this is used to prevent the error "private field is not used"
  UNUSED_RELEASE(_training_info);

  throw std::runtime_error(
    "TrainableOperationConverter: Loss operation in the model is not supported yet.");
}

} // namespace train
} // namespace compiler
} // namespace onert
