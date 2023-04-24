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

#include "SpaceToDepth.h"
#include "Utils.h"
#include "PALSpaceToDepth.h"

namespace luci_interpreter
{
namespace kernels
{

SpaceToDepth::SpaceToDepth(const Tensor *input, Tensor *output, const SpaceToDepthParams &params)
  : KernelWithParams<SpaceToDepthParams>({input}, {output}, params)
{
}

void SpaceToDepth::configure()
{
  assert(input()->shape().num_dims() == 4);
  assert(output()->element_type() == DataType::FLOAT32 ||
         output()->element_type() == DataType::U8 || output()->element_type() == DataType::S8 ||
         output()->element_type() == DataType::S32 || output()->element_type() == DataType::S64);
  assert(input()->element_type() == output()->element_type());

  const int block_size = params().block_size;
  const int32_t input_height = input()->shape().dim(1);
  const int32_t input_width = input()->shape().dim(2);
  int32_t output_height = input_height / block_size;
  int32_t output_width = input_width / block_size;

  assert(input_height == output_height * block_size);
  assert(input_width == output_width * block_size);

  Shape output_shape(4);
  output_shape.dim(0) = input()->shape().dim(0);
  output_shape.dim(1) = output_height;
  output_shape.dim(2) = output_width;
  output_shape.dim(3) = input()->shape().dim(3) * block_size * block_size;
  // TODO: enable it only if kernel with dynamic shapes
  output()->resize(output_shape);
}

void SpaceToDepth::execute() const
{
  tflite::SpaceToDepthParams op_params{};
  op_params.block_size = params().block_size;
  switch (input()->element_type())
  {
    case DataType::FLOAT32:
      luci_interpreter_pal::SpaceToDepth(op_params, getTensorShape(input()),
                                         getTensorData<float>(input()), getTensorShape(output()),
                                         getTensorData<float>(output()));
      break;
    case DataType::U8:
      luci_interpreter_pal::SpaceToDepth(op_params, getTensorShape(input()),
                                         getTensorData<uint8_t>(input()), getTensorShape(output()),
                                         getTensorData<uint8_t>(output()));
      break;
    default:
      assert(false && "Unsupported type.");
  }
}

} // namespace kernels
} // namespace luci_interpreter
