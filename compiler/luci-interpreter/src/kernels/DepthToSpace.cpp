/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "DepthToSpace.h"
#include "Utils.h"
#include <tensorflow/lite/kernels/internal/optimized/optimized_ops.h>

namespace luci_interpreter
{
namespace kernels
{

DepthToSpace::DepthToSpace(const Tensor *input, Tensor *output, const DepthToSpaceParams &params)
    : KernelWithParams<DepthToSpaceParams>({input}, {output}, params)
{
}

void DepthToSpace::configure()
{
  if (input()->shape().num_dims() != 4)
  {
    throw std::runtime_error("Invalid input num_dims.");
  }
  if (output()->element_type() != DataType::FLOAT32 && output()->element_type() != DataType::U8 &&
      output()->element_type() != DataType::S8 && output()->element_type() != DataType::S32 &&
      output()->element_type() != DataType::S64)
  {
    throw std::runtime_error("Invalid output type");
  }
  if (input()->element_type() != output()->element_type())
  {
    throw std::runtime_error("Type mismatch on input and output.");
  }
  const int block_size = params().block_size;
  const int32_t input_height = input()->shape().dim(1);
  const int32_t input_width = input()->shape().dim(2);
  const int32_t input_channels = input()->shape().dim(3);
  int32_t output_height = input_height * block_size;
  int32_t output_width = input_width * block_size;
  int32_t output_channels = input_channels / block_size / block_size;

  assert(input_height == output_height / block_size);
  assert(input_width == output_width / block_size);
  assert(input_channels == output_channels * block_size * block_size);

  Shape output_shape(4);
  output_shape.dim(0) = input()->shape().dim(0);
  output_shape.dim(1) = output_height;
  output_shape.dim(2) = output_width;
  output_shape.dim(3) = output_channels;

  output()->resize(output_shape);
}

void DepthToSpace::execute() const
{
  tflite::DepthToSpaceParams op_params;
  op_params.block_size = params().block_size;
  switch (input()->element_type())
  {
    case DataType::FLOAT32:
      tflite::optimized_ops::DepthToSpace(op_params, getTensorShape(input()),
                                          getTensorData<float>(input()), getTensorShape(output()),
                                          getTensorData<float>(output()));
      break;
    case DataType::U8:
      tflite::optimized_ops::DepthToSpace(op_params, getTensorShape(input()),
                                          getTensorData<uint8_t>(input()), getTensorShape(output()),
                                          getTensorData<uint8_t>(output()));
      break;
    default:
      throw std::runtime_error("Unsupported Type.");
  }
}

} // namespace kernels
} // namespace luci_interpreter
