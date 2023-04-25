/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

#include "kernels/Shape.h"
#include "kernels/Utils.h"

namespace luci_interpreter
{
namespace kernels
{

ShapeKernel::ShapeKernel(const Tensor *input, Tensor *output, const ShapeParams &params)
  : KernelWithParams<ShapeParams>({input}, {output}, params)
{
}

void ShapeKernel::configure()
{
  LUCI_INTERPRETER_CHECK(output()->element_type() == DataType::S32 or
                         output()->element_type() == DataType::S64);
  const auto input_shape = input()->shape();

  Shape output_shape(1);
  output_shape.dim(0) = input_shape.num_dims();
  // TODO: enable it only if kernel with dynamic shapes
  output()->resize(output_shape);
}

void ShapeKernel::execute() const
{
  switch (params().out_type)
  {
    case DataType::S32:
      evalInt<int32_t>();
      break;
    case DataType::S64:
      evalInt<int64_t>();
      break;
    default:
      assert(false && "Unsupported type.");
  }
}

template <typename T> void ShapeKernel::evalInt() const
{
  const auto input_shape = input()->shape();

  auto output_data = getTensorData<T>(output());

  for (int i = 0; i < input_shape.num_dims(); ++i)
  {
    output_data[i] = input_shape.dim(i);
  }
}

} // namespace kernels
} // namespace luci_interpreter
