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

void StaticInferer::visit(const ir::operation::Mean &op)
{
  const auto input_idx{op.getInputs().at(ir::operation::Mean::Input::INPUT)};
  const auto &input = _operands.at(input_idx);

  const auto axes_idx{op.getInputs().at(ir::operation::Mean::Input::AXES)};
  const auto &axes = _operands.at(axes_idx);

  // get mutable output operand
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

  // if input is dynamic, output also becomes dynamic
  if (input.info().isDynamic() || axes.info().isDynamic())
  {
    output.info().setDynamic();
    return;
  }

  if (!axes.isConstant())
  {
    output.info().setDynamic();
    return;
  }

  std::vector<int32_t> axes_vec;
  for (size_t i = 0; i < axes.shape().num_elements(); ++i)
  {
    switch (axes.typeInfo().type())
    {
      case ir::DataType::INT32:
      {
        axes_vec.emplace_back(reinterpret_cast<const int32_t *>(axes.data()->base())[i]);
        break;
      }
      case ir::DataType::INT64:
      {
        axes_vec.emplace_back(reinterpret_cast<const int64_t *>(axes.data()->base())[i]);
        break;
      }
      default:
        throw std::runtime_error("StaticInferer Mean: Not supported data type");
        break;
    }
  }
  const auto keep_dims = op.param().keep_dims;

  ir::Shape output_shape = inferReduceShapes(input.info().shape(), axes_vec, keep_dims);

  output.info().shape(output_shape);
}
void DynamicInferer::visit(const ir::operation::Mean &op)
{
  const auto input_idx{op.getInputs().at(ir::operation::Mean::Input::INPUT)};
  const auto &input = _tensor_registry->getITensor(input_idx);

  const auto axes_idx{op.getInputs().at(ir::operation::Mean::Input::AXES)};
  const auto &axes = _tensor_registry->getITensor(axes_idx);

  if ((!input->is_dynamic()) && !(axes->is_dynamic()))
  {
    return;
  }

  auto input_shape = input->getShape();

  auto output_ind = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_ind);

  std::vector<int32_t> axes_vec;
  for (uint32_t i = 0; i < axes->getShape().num_elements(); ++i)
  {
    const auto buffer = axes->buffer() + axes->calcOffset({i});
    switch (axes->data_type())
    {
      case ir::DataType::INT32:
      {
        axes_vec.emplace_back(*reinterpret_cast<const int32_t *>(buffer));
        break;
      }
      case ir::DataType::INT64:
      {
        axes_vec.emplace_back(*reinterpret_cast<const int64_t *>(buffer));
        break;
      }
      default:
        throw std::runtime_error("DynamicInferer Mean: Not supported data type");
        break;
    }
  }
  const auto keep_dims = op.param().keep_dims;

  ir::Shape output_shape = inferReduceShapes(input_shape, axes_vec, keep_dims);
  _dynamic_tensor_manager->applyShape(output_ind, output_shape);
  assert(output->buffer() != nullptr);
}
} // namespace shape_inference
} // namespace onert
