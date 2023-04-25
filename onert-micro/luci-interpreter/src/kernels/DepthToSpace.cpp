/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "DepthToSpace.h"
#include "Utils.h"
#include "PALDepthToSpace.h"

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
  LUCI_INTERPRETER_CHECK(input()->shape().num_dims() == 4);
  LUCI_INTERPRETER_CHECK(output()->element_type() == DataType::FLOAT32 ||
                         output()->element_type() == DataType::U8)
  LUCI_INTERPRETER_CHECK(input()->element_type() == output()->element_type())
  const int block_size = params().block_size;
  const int32_t input_height = input()->shape().dim(1);
  const int32_t input_width = input()->shape().dim(2);
  const int32_t input_channels = input()->shape().dim(3);
  int32_t output_height = input_height * block_size;
  int32_t output_width = input_width * block_size;
  int32_t output_channels = input_channels / block_size / block_size;

  LUCI_INTERPRETER_CHECK(input_height == output_height / block_size);
  LUCI_INTERPRETER_CHECK(input_width == output_width / block_size);
  LUCI_INTERPRETER_CHECK(input_channels == output_channels * block_size * block_size);

  Shape output_shape(4);
  output_shape.dim(0) = input()->shape().dim(0);
  output_shape.dim(1) = output_height;
  output_shape.dim(2) = output_width;
  output_shape.dim(3) = output_channels;

  // TODO: enable it only if kernel with dynamic shapes
  output()->resize(output_shape);
}

void DepthToSpace::execute() const
{
  tflite::DepthToSpaceParams op_params;
  op_params.block_size = params().block_size;
  switch (input()->element_type())
  {
    case DataType::FLOAT32:
      luci_interpreter_pal::DepthToSpace(op_params, getTensorShape(input()),
                                         getTensorData<float>(input()), getTensorShape(output()),
                                         getTensorData<float>(output()));
      break;
    case DataType::U8:
      luci_interpreter_pal::DepthToSpace(op_params, getTensorShape(input()),
                                         getTensorData<uint8_t>(input()), getTensorShape(output()),
                                         getTensorData<uint8_t>(output()));
      break;
    default:
      assert(false && "Unsupported Type.");
  }
}

} // namespace kernels
} // namespace luci_interpreter
