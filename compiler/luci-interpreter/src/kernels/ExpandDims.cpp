/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kernels/ExpandDims.h"
#include "kernels/Utils.h"

namespace luci_interpreter
{
namespace kernels
{

ExpandDims::ExpandDims(const Tensor *input, const Tensor *axis, Tensor *output)
  : Kernel({input, axis}, {output})
{
}

void ExpandDims::configure()
{
  int32_t axis_value;

  switch (axis()->element_type())
  {
    case loco::DataType::S32:
      axis_value = *getTensorData<int32_t>(axis());
      break;
    case loco::DataType::S64:
      axis_value = static_cast<int32_t>(*getTensorData<int64_t>(axis()));
      break;
    default:
      throw std::runtime_error("luci-intp ExpandDims Unsupported type.");
  }

  const auto input_shape = input()->shape();

  if (axis_value < 0)
  {
    axis_value += input_shape.num_dims() + 1;
  }

  LUCI_INTERPRETER_CHECK(axis_value <= input_shape.num_dims() and axis_value >= 0);

  Shape output_shape(input_shape.num_dims() + 1);
  for (int32_t i = 0; i < output_shape.num_dims(); ++i)
  {
    if (i < axis_value)
    {
      output_shape.dim(i) = input_shape.dim(i);
    }
    else if (i == axis_value)
    {
      output_shape.dim(i) = 1;
    }
    else
    {
      LUCI_INTERPRETER_CHECK(i >= 1);
      output_shape.dim(i) = input_shape.dim(i - 1);
    }
  }

  output()->resize(output_shape);
}

void ExpandDims::execute() const
{
  // Just copy input to output
  const auto *input_data = input()->data<void>();
  auto *output_data = output()->data<void>();

  const size_t element_size = getDataTypeSize(input()->element_type());
  const int32_t num_elements = input()->shape().num_elements();
  std::memcpy(output_data, input_data, num_elements * element_size);
}

} // namespace kernels
} // namespace luci_interpreter
