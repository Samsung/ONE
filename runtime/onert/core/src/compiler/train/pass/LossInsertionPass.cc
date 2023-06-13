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

#include "ir/operation/Loss.h"
#include "ir/train/operation/Loss.h"
#include "ir/train/TrainableGraph.h"
#include "ir/train/TrainingInfo.h"

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

  ir::operation::Loss::Param param;
  param.op_type = loss_info.loss_type;
  // TODO Find a way where to get `PreActivationType` information from
  param.pre_type = ir::operation::Loss::PreActivationType::SOFTMAX;

  const auto &y_pred = _trainable_graph.operands().at(loss_info.y_pred_index);
  const auto &shape = y_pred.shape();
  const auto &type_info = y_pred.typeInfo();
  auto y_true_index = _trainable_graph.addOperand(shape, type_info);
  ir::OperandIndexSequence inputs{loss_info.y_pred_index, y_true_index};

  // TODO Find correct Shape and TypeInfo
  auto output_index = _trainable_graph.addOperand(shape, type_info);
  ir::OperandIndexSequence outputs{output_index};

  auto loss_op = std::make_unique<ir::operation::Loss>(inputs, outputs, param);
  auto trainable_loss_op = std::make_unique<ir::train::operation::Loss>(*loss_op);

  _trainable_graph.addOperation(std::move(trainable_loss_op));
  // TODO Change outputs of graph
}

} // namespace pass
} // namespace train
} // namespace compiler
} // namespace onert
