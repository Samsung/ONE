/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

#include "kernels/Squeeze.h"

#include "kernels/Utils.h"

#include <stdexcept>

namespace luci_interpreter
{

namespace kernels
{

Squeeze::Squeeze(const Tensor *input, Tensor *output, const SqueezeParams &params)
    : KernelWithParams<SqueezeParams>(params), _input(input), _output(output)
{
}

void Squeeze::configure()
{
  int input_num_dims = _input->shape().num_dims();
  int num_squeeze_dims = params().squeeze_dims_count;
  const int *squeeze_dims = params().squeeze_dims;
  assert(input_num_dims <= 8);
  bool should_squeeze[8] = {false};
  int num_squeezed_dims = 0;
  if (num_squeeze_dims == 0)
  {
    for (int idx = 0; idx < input_num_dims; ++idx)
    {
      if (_input->shape().dim(idx) == 1)
      {
        should_squeeze[idx] = true;
        ++num_squeezed_dims;
      }
    }
  }
  else
  {
    for (int idx = 0; idx < num_squeeze_dims; ++idx)
    {
      int current = squeeze_dims[idx] < 0 ? squeeze_dims[idx] + input_num_dims : squeeze_dims[idx];
      assert(current >= 0 && current < input_num_dims && _input->shape().dim(current) == 1);
      if (!should_squeeze[current])
        ++num_squeezed_dims;
      should_squeeze[current] = true;
    }
  }
  Shape output_shape(input_num_dims - num_squeezed_dims);
  for (int in_idx = 0, out_idx = 0; in_idx < input_num_dims; ++in_idx)
  {
    if (!should_squeeze[in_idx])
    {
      output_shape.dim(out_idx++) = _input->shape().dim(in_idx);
    }
  }
  _output->resize(output_shape);
}

void Squeeze::execute() const
{
  size_t num_input = 1;
  size_t num_output = 1;
  for (int i = 0; i < _input->shape().num_dims(); i++)
  {
    num_input = num_input * _input->shape().dim(i);
  }
  for (int i = 0; i < _output->shape().num_dims(); i++)
  {
    num_output = num_output * _output->shape().dim(i);
  }

  assert(num_input * sizeof(getDataTypeSize(_input->element_type())) ==
         num_output * sizeof(getDataTypeSize(_output->element_type())));

  switch (_input->element_type())
  {
    case DataType::FLOAT32:
      memcpy(_output->data<float>(), _input->data<float>(), sizeof(float) * num_input);
      break;
    case DataType::U8:
      memcpy(_output->data<uint8_t>(), _input->data<uint8_t>(), sizeof(uint8_t) * num_input);
      break;
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

} // namespace kernels
} // namespace luci_interpreter
