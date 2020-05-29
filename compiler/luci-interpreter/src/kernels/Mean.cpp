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

#include "kernels/Mean.h"

#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/reference/reference_ops.h>

#include <stdexcept>

namespace luci_interpreter
{
namespace kernels
{

Mean::Mean(const Tensor *input, const Tensor *reduction_indices, Tensor *output,
           const ReducerParams &params)
    : KernelWithParams<ReducerParams>(params), _input(input), _reduction_indices(reduction_indices),
      _output(output)
{
}

void Mean::configure()
{
  assert(_input->element_type() == _output->element_type());
  assert(_reduction_indices->element_type() == DataType::S32);
  const Shape input_shape = _input->shape();
  int input_num_dims = input_shape.num_dims();

  const auto *axis = getTensorData<int32_t>(_reduction_indices);
  int num_axis = _reduction_indices->shape().num_elements();
  assert(num_axis <= 4);

  if (input_num_dims == 0)
  {
    _output->resize(Shape(0));
    return;
  }

  if (_params.keep_dims)
  {
    Shape output_shape(input_num_dims);
    for (int idx = 0; idx < input_num_dims; ++idx)
    {
      bool is_axis = false;
      for (int axis_idx = 0; axis_idx < num_axis; ++axis_idx)
      {
        if (axis[axis_idx] == idx || axis[axis_idx] + input_num_dims == idx)
        {
          is_axis = true;
          break;
        }
      }
      if (is_axis)
      {
        output_shape.dim(idx) = 1;
      }
      else
      {
        output_shape.dim(idx) = input_shape.dim(idx);
      }
    }
    _output->resize(output_shape);
    return;
  }
  else
  {
    // Calculates size of reducing axis.
    int num_reduce_axis = num_axis;
    for (int i = 0; i < num_axis; ++i)
    {
      int current = axis[i];
      if (current < 0)
      {
        current += input_num_dims;
      }
      assert(current >= 0 && current < input_num_dims);
      for (int j = 0; j < i; j++)
      {
        int previous = axis[j];
        if (previous < 0)
        {
          previous += input_num_dims;
        }
        if (current == previous)
        {
          --num_reduce_axis;
          break;
        }
      }
    }
    // Determines output dimensions.
    Shape output_shape(input_num_dims - num_reduce_axis);
    int num_skip_axis = 0;
    for (int idx = 0; idx < input_num_dims; ++idx)
    {
      bool is_axis = false;
      for (int axis_idx = 0; axis_idx < num_axis; ++axis_idx)
      {
        if (axis[axis_idx] == idx || axis[axis_idx] + input_num_dims == idx)
        {
          ++num_skip_axis;
          is_axis = true;
          break;
        }
      }
      if (!is_axis)
      {
        output_shape.dim(idx - num_skip_axis) = input_shape.dim(idx);
      }
    }
    _output->resize(output_shape);
  }
}

void Mean::execute() const
{
  switch (_input->element_type())
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

void Mean::evalFloat() const
{
  const Shape input_shape = _input->shape();
  int input_num_dims = input_shape.num_dims();
  const auto *axis = getTensorData<int32_t>(_reduction_indices);
  int num_axis = _reduction_indices->shape().num_elements();

  // TODO
  // Declare these temporary tensors somewhere else?
  Tensor *temp_index = new Tensor(DataType::S32, Shape(input_num_dims), {}, "");
  Tensor *resolved_axis = new Tensor(DataType::S32, Shape(num_axis), {}, "");
  Tensor *temp_sum = new Tensor(DataType::FLOAT32, Shape(_output->shape().num_elements()), {}, "");

  tflite::MeanParams params{};
  params.axis_count = num_axis;
  // Resolve axis
  for (int i = 0; i < num_axis; ++i)
  {
    params.axis[i] = static_cast<int16>(axis[i]);
  }
  for (int i = num_axis; i < 4; ++i)
  {
    params.axis[i] = 1;
  }

  if (_params.keep_dims && input_num_dims == 4 && params.axis_count == 2 &&
      ((params.axis[0] == 1 && params.axis[1] == 2) ||
       (params.axis[0] == 2 && params.axis[1] == 1)))
  {
    tflite::reference_ops::Mean(params, getTensorShape(_input), getTensorData<float>(_input),
                                getTensorShape(_output), getTensorData<float>(_output));
  }
  else
  {
    tflite::reference_ops::Mean(getTensorData<float>(_input), getTensorShape(_input).DimsData(),
                                _input->shape().num_dims(), getTensorData<float>(_output),
                                getTensorShape(_output).DimsData(), _output->shape().num_dims(),
                                axis, num_axis, _params.keep_dims, getTensorData<int>(temp_index),
                                getTensorData<int>(resolved_axis), getTensorData<float>(temp_sum));
  }
}

void Mean::evalQuantized() const
{
  const Shape input_shape = _input->shape();
  int input_num_dims = input_shape.num_dims();
  const auto *axis = getTensorData<int32_t>(_reduction_indices);
  int num_axis = _reduction_indices->shape().num_elements();

  // TODO
  // Declare these temporary tensors somewhere else?
  Tensor *temp_index = new Tensor(DataType::S32, Shape({input_num_dims}), {}, "");
  Tensor *resolved_axis = new Tensor(DataType::S32, Shape({num_axis}), {}, "");
  Tensor *temp_sum = new Tensor(DataType::S32, Shape({_output->shape().num_elements()}), {}, "");

  tflite::MeanParams params{};
  params.axis_count = num_axis;
  // Resolve axis
  for (int i = 0; i < num_axis; ++i)
  {
    params.axis[i] = static_cast<int16>(axis[i]);
  }
  for (int i = num_axis; i < 4; ++i)
  {
    params.axis[i] = 1;
  }

  if (_params.keep_dims && input_num_dims == 4 && params.axis_count == 2 &&
      ((params.axis[0] == 1 && params.axis[1] == 2) ||
       (params.axis[0] == 2 && params.axis[1] == 1)))
  {
    tflite::reference_ops::Mean(params, getTensorShape(_input), getTensorData<uint8_t>(_input),
                                _input->zero_point(), _input->scale(), getTensorShape(_output),
                                getTensorData<uint8_t>(_output), _output->zero_point(),
                                _output->scale());
  }
  else if (_input->zero_point() == _output->zero_point() && _input->scale() == _output->scale())
  {
    tflite::reference_ops::Mean(getTensorData<uint8_t>(_input), getTensorShape(_input).DimsData(),
                                _input->shape().num_dims(), getTensorData<uint8_t>(_output),
                                getTensorShape(_output).DimsData(), _output->shape().num_dims(),
                                axis, num_axis, _params.keep_dims, getTensorData<int>(temp_index),
                                getTensorData<int>(resolved_axis), getTensorData<int>(temp_sum));
  }
  else
  {
    tflite::reference_ops::QuantizedMeanOrSum<>(
        getTensorData<uint8_t>(_input), _input->zero_point(), _input->scale(),
        getTensorShape(_input).DimsData(), _input->shape().num_dims(),
        getTensorData<uint8_t>(_output), _output->zero_point(), _output->scale(),
        getTensorShape(_output).DimsData(), _output->shape().num_dims(), axis, num_axis,
        _params.keep_dims, getTensorData<int>(temp_index), getTensorData<int>(resolved_axis),
        getTensorData<int>(temp_sum),
        /*compute_sum=*/false);
  }
}

} // namespace kernels
} // namespace luci_interpreter
