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

#include "SpaceToDepth.h"
#include "Utils.h"
#include <tensorflow/lite/kernels/internal/optimized/optimized_ops.h>

namespace luci_interpreter
{
namespace kernels
{

SpaceToDepth::SpaceToDepth(const Tensor *input, Tensor *output, const SpaceToDepthParams &params)
    : KernelWithParams<SpaceToDepthParams>(params), _input(input), _output(output)
{
}

void SpaceToDepth::configure()
{
  assert(_input->shape().num_dims() == 4);
  assert(_output->element_type() == DataType::FLOAT32 || _output->element_type() == DataType::U8 ||
         _output->element_type() == DataType::S8 || _output->element_type() == DataType::S32 ||
         _output->element_type() == DataType::S64);
  assert(_input->element_type() == _output->element_type());

  const int block_size = params().block_size;
  const int32_t input_height = _input->shape().dim(1);
  const int32_t input_width = _input->shape().dim(2);
  int32_t output_height = input_height / block_size;
  int32_t output_width = input_width / block_size;

  assert(input_height == output_height * block_size);
  assert(input_width == output_width * block_size);

  Shape output_shape(4);
  output_shape.dim(0) = _input->shape().dim(0);
  output_shape.dim(1) = output_height;
  output_shape.dim(2) = output_width;
  output_shape.dim(3) = _input->shape().dim(3) * block_size * block_size;

  _output->resize(output_shape);
}

void SpaceToDepth::execute() const
{
  tflite::SpaceToDepthParams op_params{};
  op_params.block_size = params().block_size;
  switch (_input->element_type())
  {
    case DataType::FLOAT32:
      tflite::optimized_ops::SpaceToDepth(op_params, getTensorShape(_input),
                                          getTensorData<float>(_input), getTensorShape(_output),
                                          getTensorData<float>(_output));
      break;
    case DataType::U8:
      tflite::optimized_ops::SpaceToDepth(op_params, getTensorShape(_input),
                                          getTensorData<uint8_t>(_input), getTensorShape(_output),
                                          getTensorData<uint8_t>(_output));
      break;
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

} // namespace kernels
} // namespace luci_interpreter
