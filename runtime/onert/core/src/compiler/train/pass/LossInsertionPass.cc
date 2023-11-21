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

  ir::operation::Loss::Param param;
  param.op_type = loss_info.type;

  if (_trainable_graph.getOutputs().size() != 1)
  {
    throw std::runtime_error("LossInsertionPass: Not supported multiple outputs");
  }

  // TODO Consider SparseCategoricalCrossentropy y_true shape
  //      SparseCategoricalCrossentropy loss has a different y_true shape than y_pred.

  ir::OperandIndex input_index;

  _trainable_graph.operations().iterate([&](const ir::OperationIndex &, const ir::IOperation &op) {
    if (op.opcode() == ir::OpCode::Softmax && param.op_type == ir::operation::Loss::Type::CATEGORICAL_CROSSENTROPY)
    {
      input_index = op.getInputs().at(0);
      return;
    }
  });

  if (!input_index.valid())
  {
    input_index = _trainable_graph.getOutputs().at(0);
  }

  const auto &input = _trainable_graph.operands().at(input_index);
  auto y_true_index = _trainable_graph.addOperand(input.shape(), input.typeInfo());

  // TODO Consider Reduction
  //      Some types of Reduction have the same shape y_true and output.

  const ir::TypeInfo float_op(ir::DataType::FLOAT32);
  auto output_index = _trainable_graph.addOperand(ir::Shape{1}, float_op);

  ir::OperandIndexSequence inputs{input_index, y_true_index};
  ir::OperandIndexSequence outputs{output_index};
  auto loss_op = std::make_unique<ir::operation::Loss>(inputs, outputs, param);
  auto trainable_loss_op = std::make_unique<ir::train::operation::Loss>(*loss_op);
  _trainable_graph.addOperation(std::move(trainable_loss_op));

  _trainable_graph.addInput(y_true_index);

  // TODO Add loss as many as output size
  _trainable_graph.addLoss(output_index, ir::IOIndex{0});
}

} // namespace pass
} // namespace train
} // namespace compiler
} // namespace onert
