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
// helper function
template <typename T> ir::Shape inferRangeShape(T start_val, T limit_val, T delta_val)
{
  ir::Shape out_shape(static_cast<int>(1));

  out_shape.dim(0) =
      (std::is_integral<T>::value
           ? ((std::abs(*start_val - *limit_val) + std::abs(*delta_val) - 1) / std::abs(*delta_val))
           : std::ceil(std::abs((*start_val - *limit_val) / *delta_val)));
  return out_shape;
}

// StaticInferer at compilation time
void StaticInferer::visit(const ir::operation::Range &op)
{
  const auto start_idx{op.getInputs().at(ir::operation::Range::Input::START)};
  const auto limit_idx{op.getInputs().at(ir::operation::Range::Input::LIMIT)};
  const auto delta_idx{op.getInputs().at(ir::operation::Range::Input::DELTA)};
  const auto &start_op = _operands.at(start_idx);
  const auto &limit_op = _operands.at(limit_idx);
  const auto &delta_op = _operands.at(delta_idx);

  // get mutable output operand
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);
  // if any input is dynamic, output also becomes dynamic
  if (start_op.info().isDynamic() || limit_op.info().isDynamic() || delta_op.info().isDynamic())
  {
    output.info().setDynamic();
    return;
  }
}

// DynamicInferer at execution time
void DynamicInferer::visit(const ir::operation::Range &op)
{
  // check if output is not dynamic
  auto output_ind = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_ind);

  // from op, access the buffer of second input to read new shape
  auto start_idx = op.getInputs().at(ir::operation::Range::Input::START);
  auto start_tensor = _tensor_registry->getITensor(start_idx);

  auto limit_idx = op.getInputs().at(ir::operation::Range::Input::LIMIT);
  auto limit_tensor = _tensor_registry->getITensor(limit_idx);

  auto delta_idx = op.getInputs().at(ir::operation::Range::Input::DELTA);
  auto delta_tensor = _tensor_registry->getITensor(delta_idx);

  if (!start_tensor->is_dynamic() && !limit_tensor->is_dynamic() && !delta_tensor->is_dynamic())
    return;

  ir::Shape new_shape;
  if (output->data_type() == ir::DataType::FLOAT32)
  {
    new_shape = inferRangeShape<float *>( reinterpret_cast<float *>(start_tensor->buffer()),
                                          reinterpret_cast<float *>(limit_tensor->buffer()),
                                          reinterpret_cast<float *>(delta_tensor->buffer()));
  }
  else if (output->data_type() == ir::DataType::INT32)
  {
    new_shape = inferRangeShape<int32_t *>( reinterpret_cast<int32_t *>(start_tensor->buffer()),
                                            reinterpret_cast<int32_t *>(limit_tensor->buffer()),
                                            reinterpret_cast<int32_t *>(delta_tensor->buffer()));
  }
  _dynamic_tensor_manager->applyShape(output_ind, new_shape);
  assert(output->buffer() != nullptr);
}
} // namespace shape_inference
} // namespace onert
