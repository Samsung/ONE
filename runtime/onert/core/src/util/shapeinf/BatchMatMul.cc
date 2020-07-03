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

void StaticInferer::visit(const ir::operation::BatchMatMul &op)
{
  const auto lhs_index = op.getInputs().at(ir::operation::BatchMatMul::Input::LHS);
  const auto rhs_index = op.getInputs().at(ir::operation::BatchMatMul::Input::RHS);
  const auto output_index = op.getOutputs().at(0);
  const auto lhs = _operands.at(lhs_index);
  const auto rhs = _operands.at(rhs_index);
  auto &output = _operands.at(output_index);

  if (lhs.info().isDynamic() || rhs.info().isDynamic())
  {
    output.info().setDynamic();
    return;
  }

  auto new_shape = inferBatchMatMulShape(lhs.shape(), rhs.shape(), op.param());
  output.info().shape(new_shape);
}

void DynamicInferer::visit(const ir::operation::BatchMatMul &op)
{
  const auto lhs_index = op.getInputs().at(ir::operation::BatchMatMul::Input::LHS);
  const auto rhs_index = op.getInputs().at(ir::operation::BatchMatMul::Input::RHS);
  auto lhs = _tensor_registry->getITensor(lhs_index);
  auto rhs = _tensor_registry->getITensor(rhs_index);

  if (!lhs->is_dynamic() && !rhs->is_dynamic())
    return;

  const auto output_index = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_index);

  auto lhs_shape = lhs->getShape();
  auto rhs_shape = rhs->getShape();
  // TODO

  auto new_shape = inferBatchMatMulShape(lhs_shape, rhs_shape, op.param());
  _dynamic_tensor_manager->applyShape(output_index, new_shape);
}

} // namespace shape_inference
} // namespace onert
