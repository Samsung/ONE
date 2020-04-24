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

ir::Shape inferConcatShape(const Shapes &in_shapes, const ir::operation::Concat::Param &param)
{
  const int32_t concat_axis = param.axis;
  const auto &first_in_shape = in_shapes[0];

  // Check that all shapes are equal except for concat axis dimension
  for (const auto &in_shape : in_shapes)
  {
    assert(in_shape.rank() == first_in_shape.rank());
    for (int64_t dim_idx = 0; dim_idx < in_shape.rank(); ++dim_idx)
      assert(dim_idx == concat_axis || in_shape.dim(dim_idx) == first_in_shape.dim(dim_idx));
  }

  // Calculate output shape
  ir::Shape out_shape(first_in_shape);
  out_shape.dim(concat_axis) = 0;
  for (const auto &in_shape : in_shapes)
    out_shape.dim(concat_axis) += in_shape.dim(concat_axis);
  return out_shape;
}

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
  // check if output is not dynamic
  auto output_ind = op.getOutputs().at(0);
  auto *output = _tensor_registry->getITensor(output_ind);
  if (!output->is_dynamic())
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
    auto *first_input = _tensor_registry->getITensor(first_input_ind);

    for (auto input_ind : op.getInputs())
    {
      auto *input = _tensor_registry->getITensor(input_ind);
      if (input != first_input && !isConcatible(first_input, input, op.param().axis))
        throw std::runtime_error("input shapes does not matched for concat");
    }
  }

  // getting output shape
  onert::shape_inference::Shapes in_shapes;
  for (auto input_ind : op.getInputs())
  {
    auto *input = _tensor_registry->getITensor(input_ind);
    ir::Shape shape = getShape(input);

    in_shapes.emplace_back(shape);
  }

  auto output_shape = onert::shape_inference::inferConcatShape(in_shapes, op.param());

  // set output shape and output buffer
  setShape(output, output_shape);

  // assert(output->buffer() == nullptr);
  _dynamic_tensor_manager->allocate(output_ind, output_shape);
}

} // namespace shape_inference
} // namespace onert
