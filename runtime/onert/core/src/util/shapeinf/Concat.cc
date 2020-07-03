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

#include <cassert>

namespace onert
{
namespace shape_inference
{

void StaticInferer::visit(const ir::operation::Concat &op)
{
  const auto input_count = op.getInputs().size();

  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

  Shapes input_shapes;
  for (uint32_t i = 0; i < input_count; i++)
  {
    const auto input_idx{op.getInputs().at(i)};
    const auto &input = _operands.at(input_idx);

    if (input.info().isDynamic())
    {
      output.info().setDynamic();
      return;
    }

    input_shapes.emplace_back(input.shape());
  }

  ir::Shape out_shape = inferConcatShape(input_shapes, op.param());

  // re-sizing output shape
  output.info().shape(out_shape);
}

void DynamicInferer::visit(const ir::operation::Concat &op)
{
  /*
    The state after compilation (satic shape inference) could be one of the following:

              inputs                  output        execution-time shape inf required
      ------------------------------------------    ---------------------------------
      case 1) all static              static         X
      case 2) at least on is dynamic  dynamic        O

    Then nnfw_apply_tensorinf() could change one or both inputs dynamic.
    So, in this method, we have one more state and we have to re-calculate shape for this shape.

      case 3) at least on is dynamic  static         O

    So, only when all inputs are static, we can skip dynamic shape inference.
  */
  bool all_static = true;
  for (auto input_ind : op.getInputs())
  {
    auto input = _tensor_registry->getITensor(input_ind);
    if (input->is_dynamic())
    {
      all_static = false;
      break;
    }
  }

  if (all_static)
    return;

  // sanity check
  {
    auto isConcatible = [](const backend::ITensor *input1, const backend::ITensor *input2,
                           int32_t axis) {
      if (input1->num_dimensions() != input2->num_dimensions())
        return false;

      for (size_t i = 0; i < input1->num_dimensions(); i++)
      {
        auto positive_axis = (axis >= 0) ? axis : axis + input1->num_dimensions();

        if (i != positive_axis)
          if (input1->dimension(i) != input2->dimension(i))
            return false;
      }

      return true;
    };

    auto first_input_ind = op.getInputs().at(0);
    auto first_input = _tensor_registry->getITensor(first_input_ind);

    for (auto input_ind : op.getInputs())
    {
      auto input = _tensor_registry->getITensor(input_ind);
      if (input != first_input && !isConcatible(first_input.get(), input.get(), op.param().axis))
        throw std::runtime_error("input shapes does not matched for concat");
    }
  }

  // getting output shape
  onert::shape_inference::Shapes in_shapes;
  for (auto input_ind : op.getInputs())
  {
    auto input = _tensor_registry->getITensor(input_ind);
    ir::Shape shape = input->getShape();

    in_shapes.emplace_back(shape);
  }

  auto output_ind = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_ind);
  auto output_shape = onert::shape_inference::inferConcatShape(in_shapes, op.param());

  _dynamic_tensor_manager->applyShape(output_ind, output_shape);
}

} // namespace shape_inference
} // namespace onert
