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

ir::Shape inferSqueezeShape(const ir::Shape &in_shape, const ir::operation::Squeeze::Param &param)
{
  const int ndims = param.ndim;
  const int *squeeze_dims = param.dims;
  bool should_squeeze[8] = {false};
  int num_squeezed_dims = 0;
  int shape_rank = in_shape.rank();
  if (ndims == 0)
  {
    for (int idx = 0; idx < shape_rank; ++idx)
    {
      if (in_shape.dim(idx) == 1)
      {
        should_squeeze[idx] = true;
        ++num_squeezed_dims;
      }
    }
  }
  else
  {
    for (int idx = 0; idx < ndims; ++idx)
    {
      int current = squeeze_dims[idx];
      if (current < 0)
      {
        current += shape_rank;
      }

      if (!(current >= 0 && current < shape_rank && in_shape.dim(current) == 1))
      {
        throw std::runtime_error(
            "The following conditions must be met: 0 <= dim < Shape rank, dim == 1");
      }

      if (!should_squeeze[current])
      {
        ++num_squeezed_dims;
      }
      should_squeeze[current] = true;
    }
  }

  // Set output shape.
  ir::Shape out_shape(shape_rank - num_squeezed_dims);
  for (int in_idx = 0, out_idx = 0; in_idx < shape_rank; ++in_idx)
  {
    if (!should_squeeze[in_idx])
    {
      out_shape.dim(out_idx++) = in_shape.dim(in_idx);
    }
  }

  return out_shape;
}

void StaticInferer::visit(const ir::operation::Squeeze &op)
{
  const auto input_idx{op.getInputs().at(ir::operation::Squeeze::Input::INPUT)};
  const auto &input = _operands.at(input_idx);

  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

  if (input.info().isDynamic())
  {
    output.info().setDynamic();
    return;
  }

  // Squeeze output shpae
  ir::Shape new_shape = inferSqueezeShape(input.info().shape(), op.param());
  output.info().shape(new_shape);
}

void DynamicInferer::visit(const ir::operation::Squeeze &op)
{
  const auto input_idx{op.getInputs().at(ir::operation::Squeeze::Input::INPUT)};
  const auto &input = _tensor_registry->getITensor(input_idx);

  if (!input->is_dynamic())
  {
    return;
  }

  auto input_shape = getShape(input.get());

  // Squeeze output shpae
  ir::Shape new_shape = inferSqueezeShape(input_shape, op.param());

  auto output_ind = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_ind);

  _dynamic_tensor_manager->applyShape(output_ind, new_shape);
  assert(output->buffer() != nullptr);
}

} // namespace shape_inference
} // namespace onert
