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

void StaticInferer::visit(const ir::operation::Split &op)
{
  const auto input_idx{op.getInputs().at(0)};
  const auto &input = _operands.at(input_idx);

  const auto axis = op.param().axis;
  const auto num_splits = op.param().num_splits;

  if (input.info().isDynamic())
  {
    for (int out_tensor_idx = 0; out_tensor_idx < num_splits; out_tensor_idx++)
    {
      const auto output_idx = op.getOutputs().at(out_tensor_idx);
      ir::Operand &output = _operands.at(output_idx);
      output.info().setDynamic();
    }
    return;
  }

  const auto rank = input.info().shape().rank();
  auto axis_resolved = axis < 0 ? axis + rank : axis;

  assert(0 <= axis_resolved && axis_resolved < rank);

  ir::Shape new_shape = inferSplitShape(input.info().shape(), axis_resolved, num_splits);
  auto output_teonsors = op.getOutputs();
  for (auto output_idx : output_teonsors)
  {
    ir::Operand &output = _operands.at(output_idx);
    output.info().shape(new_shape);
  }
}

void DynamicInferer::visit(const ir::operation::Split &op)
{
  const auto input_idx{op.getInputs().at(ir::operation::Split::Input::INPUT)};
  const auto &input = _tensor_registry->getITensor(input_idx);

  if (!input->is_dynamic())
  {
    return;
  }

  auto input_shape = input->getShape();

  const auto axis = op.param().axis;
  const auto num_splits = op.param().num_splits;
  const auto rank = input_shape.rank();
  auto axis_resolved = axis < 0 ? axis + rank : axis;

  assert(0 <= axis_resolved && axis_resolved < rank);

  ir::Shape new_shape = inferSplitShape(input_shape, axis_resolved, num_splits);
  for (int out_tensor_idx = 0; out_tensor_idx < num_splits; out_tensor_idx++)
  {
    auto output_ind = op.getOutputs().at(out_tensor_idx);
    auto output = _tensor_registry->getITensor(output_ind);

    _dynamic_tensor_manager->applyShape(output_ind, new_shape);
    assert(output->buffer() != nullptr);
  }
}
} // namespace shape_inference
} // namespace onert
