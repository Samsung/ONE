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

#include "LossInsertionPass.h"

#include "ir/train/TrainableGraph.h"
#include "ir/train/operation/Loss.h"

namespace onert
{
namespace compiler
{
namespace train
{
namespace pass
{

void LossInsertionPass::run()
{
  const auto &loss_info = _training_info->lossInfo();

  if (_trainable_graph.getOutputs().size() != 1)
  {
    throw std::runtime_error("LossInsertionPass: Not supported multiple outputs");
  }

  // TODO Consider SparseCategoricalCrossentropy y_true shape
  //      SparseCategoricalCrossentropy loss has a different y_true shape than y_pred.

  // TODO Implement Loop [0, getOutputs().size())
  //      index: a loop index
  const auto index = 0;
  const auto &y_pred_index = _trainable_graph.getOutputs().at(index);
  const auto &y_pred = _trainable_graph.operands().at(y_pred_index);
  auto y_true_index = _trainable_graph.addOperand(y_pred.shape(), y_pred.typeInfo());
  ir::OperandIndexSequence inputs{y_pred_index, y_true_index};

  ir::Shape output_shape;
  if (loss_info.reduction_type == ir::train::LossReductionType::Sum ||
      loss_info.reduction_type == ir::train::LossReductionType::SumOverBatchSize)
  {
    output_shape = ir::Shape{1};
  }
  else
  {
    throw std::runtime_error("LossInsertionPass: Not supported reduction type");
  }

  const ir::TypeInfo float_op(ir::DataType::FLOAT32);
  auto output_index = _trainable_graph.addOperand(output_shape, float_op);
  ir::OperandIndexSequence outputs{output_index};

  auto loss_op = std::make_unique<ir::operation::Loss>(inputs, outputs);
  auto trainable_loss_op = std::make_unique<ir::train::operation::Loss>(*loss_op, loss_info);

  _trainable_graph.addOperation(std::move(trainable_loss_op));

  _trainable_graph.addInput(y_true_index);

  // TODO Add loss as many as output size
  _trainable_graph.addLoss(output_index, ir::IOIndex{index});
}

} // namespace pass
} // namespace train
} // namespace compiler
} // namespace onert
