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

#include "kernels/L2Normalize.h"
#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/optimized/optimized_ops.h>

#include <stdexcept>

namespace luci_interpreter
{

namespace kernels
{

L2Normalize::L2Normalize(const Tensor *input, Tensor *output) : _input(input), _output(output) {}

void L2Normalize::configure()
{
  assert(_input->shape().num_dims() <= 4);
  assert(_output->element_type() == DataType::FLOAT32 || _output->element_type() == DataType::U8);
  assert(_input->element_type() == _output->element_type());
  if (_output->element_type() == DataType::U8)
  {
    assert(_output->scale() == (1. / 128.));
    assert(_output->zero_point() == 128);
  }
  int dims = _input->shape().num_dims();
  Shape output_shape(dims);
  for (int i = 0; i < dims; i++)
    output_shape.dim(i) = _input->shape().dim(i);
  _output->resize(output_shape);
}

void L2Normalize::execute() const
{
  switch (_output->element_type())
  {
    case DataType::FLOAT32:
      evalFloat();
      break;
    case DataType::U8:
      evalQuantized();
      break;
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

void L2Normalize::evalFloat() const
{
  tflite::L2NormalizationParams op_params{};
  op_params.input_zero_point = 0;
  tflite::optimized_ops::L2Normalization(op_params, getTensorShape(_input),
                                         getTensorData<float>(_input), getTensorShape(_output),
                                         getTensorData<float>(_output));
}

void L2Normalize::evalQuantized() const
{
  tflite::L2NormalizationParams op_params{};
  op_params.input_zero_point = _input->zero_point();
  tflite::optimized_ops::L2Normalization(op_params, getTensorShape(_input),
                                         getTensorData<uint8_t>(_input), getTensorShape(_output),
                                         getTensorData<uint8_t>(_output));
}

} // namespace kernels
} // namespace luci_interpreter
