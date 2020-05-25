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

void StaticInferer::visit(const ir::operation::Add &op)
{
  const auto lhs_idx{op.getInputs().at(ir::operation::Add::Input::LHS)};
  const auto &lhs = _operands.at(lhs_idx);
  const auto rhs_idx{op.getInputs().at(ir::operation::Add::Input::RHS)};
  const auto &rhs = _operands.at(rhs_idx);
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

  if (lhs.info().isDynamic() || rhs.info().isDynamic())
  {
    output.info().setDynamic();
    return;
  }

  // re-sizing output shape
  ir::Shape new_shape = inferEltwiseShape(lhs.info().shape(), rhs.info().shape());
  output.info().shape(new_shape);
}

void DynamicInferer::visit(const ir::operation::Add &op)
{
  // check if output is not dynamic
  auto output_ind = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_ind);
  if (!output->is_dynamic())
    return;

  // getting output shape
  const auto lhs_ind{op.getInputs().at(ir::operation::Add::Input::LHS)};
  auto lhs = _tensor_registry->getITensor(lhs_ind);
  auto lhs_shape = getShape(lhs.get());

  const auto rhs_ind{op.getInputs().at(ir::operation::Add::Input::RHS)};
  auto rhs = _tensor_registry->getITensor(rhs_ind);
  auto rhs_shape = getShape(rhs.get());

  // set output shape and output buffer
  ir::Shape new_shape = inferEltwiseShape(lhs_shape, rhs_shape);
  setShape(output.get(), new_shape);

  _dynamic_tensor_manager->allocate(output_ind, new_shape);
  assert(output->buffer() != nullptr);
}

} // namespace shape_inference
} // namespace onert
