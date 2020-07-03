/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "util/ShapeInference.h"

namespace onert
{
namespace shape_inference
{

void StaticInferer::visit(const ir::operation::Select &op)
{
  const auto input_cond_idx{op.getInputs().at(ir::operation::Select::Input::CONDITION)};
  const auto &input_cond = _operands.at(input_cond_idx);

  const auto input_true_idx{op.getInputs().at(ir::operation::Select::Input::INPUT_TRUE)};
  const auto &input_true = _operands.at(input_true_idx);

  const auto input_false_idx{op.getInputs().at(ir::operation::Select::Input::INPUT_FALSE)};
  const auto &input_false = _operands.at(input_false_idx);

  auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

  if (input_cond.info().isDynamic() || input_true.info().isDynamic() ||
      input_false.info().isDynamic())
  {
    output.info().setDynamic();
    return;
  }

  // Select output shpae
  ir::Shape new_shape = inferSelectShape(input_cond.info().shape(), input_true.info().shape(),
                                         input_false.info().shape());
  output.info().shape(new_shape);
}

void DynamicInferer::visit(const ir::operation::Select &op)
{
  const auto input_cond_idx = op.getInputs().at(ir::operation::Select::Input::CONDITION);
  const auto &input_cond = _tensor_registry->getITensor(input_cond_idx);

  const auto input_true_idx = op.getInputs().at(ir::operation::Select::Input::INPUT_TRUE);
  const auto &input_true = _tensor_registry->getITensor(input_true_idx);

  const auto input_false_idx = op.getInputs().at(ir::operation::Select::Input::INPUT_FALSE);
  const auto &input_false = _tensor_registry->getITensor(input_false_idx);

  if ((!input_cond->is_dynamic()) && (!input_true->is_dynamic()) && (!input_false->is_dynamic()))
  {
    return;
  }

  auto input_cond_shape = input_cond->getShape();
  auto input_true_shape = input_true->getShape();
  auto input_false_shape = input_false->getShape();

  // Select output shpae
  ir::Shape new_shape = inferSelectShape(input_cond_shape, input_true_shape, input_false_shape);

  auto output_ind = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_ind);

  _dynamic_tensor_manager->applyShape(output_ind, new_shape);
  assert(output->buffer() != nullptr);
}

} // namespace shape_inference
} // namespace onert
