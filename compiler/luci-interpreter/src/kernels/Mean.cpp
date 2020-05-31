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

static void ResolveAxis(const int *axis_data, int num_axis, tflite::MeanParams *params)
{
  params->axis_count = num_axis;
  for (int i = 0; i < num_axis; ++i)
  {
    params->axis[i] = static_cast<int16>(axis_data[i]);
  }
  for (int i = num_axis; i < 4; ++i)
  {
    params->axis[i] = 1;
  }
}

// Returns the number of axis that will be reduced. Removes duplicates.
static int GetAxisReductionCount(const int *axis_data, int num_axis, int input_num_dims)
{
  int reduction_count = num_axis;
  for (int i = 0; i < num_axis; ++i)
  {
    int current = axis_data[i] >= 0 ? axis_data[i] : axis_data[i] + input_num_dims;
    assert(current >= 0 && current < input_num_dims);
    for (int j = 0; j < i; j++)
    {
      int previous = axis_data[j] >= 0 ? axis_data[j] : axis_data[j] + input_num_dims;
      // This checks for duplicate axis
      if (current == previous)
      {
        --reduction_count;
        break;
      }
    }
  }
  return reduction_count;
}

static void ResizeOutputTensor(const Shape &input_shape, int input_num_dims, const int *axis_data,
                               int num_axis, bool keep_dims, Tensor *output)
{
  if (input_num_dims == 0)
  {
    output->resize(Shape(0));
    return;
  }

  if (keep_dims)
  {
    Shape output_shape(input_num_dims);
    for (int idx = 0; idx < input_num_dims; ++idx)
    {
      bool is_axis = false;
      for (int axis_idx = 0; axis_idx < num_axis; ++axis_idx)
      {
        if (axis_data[axis_idx] == idx || axis_data[axis_idx] + input_num_dims == idx)
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
    output->resize(output_shape);
  }
  else
  {
    int num_reduce_axis = GetAxisReductionCount(axis_data, num_axis, input_num_dims);
    Shape output_shape(input_num_dims - num_reduce_axis);
    int num_skip_axis = 0;
    for (int idx = 0; idx < input_num_dims; ++idx)
    {
      bool is_axis = false;
      for (int axis_idx = 0; axis_idx < num_axis; ++axis_idx)
      {
        if (axis_data[axis_idx] == idx || axis_data[axis_idx] + input_num_dims == idx)
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
    output->resize(output_shape);
  }
}

Mean::Mean(const Tensor *input, const Tensor *axis, Tensor *output, const ReducerParams &params)
    : KernelWithParams<ReducerParams>(params), _input(input), _axis(axis), _output(output)
{
}

void Mean::configure()
{
  assert(_input->element_type() == _output->element_type());
  assert(_axis->element_type() == DataType::S32);
  const Shape &input_shape = _input->shape();
  int input_num_dims = input_shape.num_dims();

  const auto *axis_data = getTensorData<int32_t>(_axis);
  int num_axis = _axis->shape().num_elements();
  assert(num_axis <= 4);

  tflite::MeanParams params{};
  ResolveAxis(axis_data, num_axis, &params);
  const bool need_temporaries =
      !(_params.keep_dims && input_num_dims == 4 && params.axis_count == 2 &&
        ((params.axis[0] == 1 && params.axis[1] == 2) ||
         (params.axis[0] == 2 && params.axis[1] == 1)));
  if (need_temporaries)
  {
    _temp_index =
        std::make_unique<Tensor>(DataType::S32, Shape(input_num_dims), AffineQuantization{}, "");
    _resolved_axis =
        std::make_unique<Tensor>(DataType::S32, Shape(num_axis), AffineQuantization{}, "");
    _temp_sum = std::make_unique<Tensor>(
        _input->element_type(), Shape(_output->shape().num_elements()), AffineQuantization{}, "");
  }
  ResizeOutputTensor(input_shape, input_num_dims, axis_data, num_axis, _params.keep_dims, _output);
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
  const Shape &input_shape = _input->shape();
  int input_num_dims = input_shape.num_dims();
  const auto *axis_data = getTensorData<int32_t>(_axis);
  int num_axis = _axis->shape().num_elements();

  tflite::MeanParams params{};
  ResolveAxis(axis_data, num_axis, &params);

  // Defer to specialized implementation for 4D Mean across axes 1 & 2.
  if (_params.keep_dims && input_num_dims == 4 && params.axis_count == 2 &&
      ((params.axis[0] == 1 && params.axis[1] == 2) ||
       (params.axis[0] == 2 && params.axis[1] == 1)))
  {
    tflite::reference_ops::Mean(params, getTensorShape(_input), getTensorData<float>(_input),
                                getTensorShape(_output), getTensorData<float>(_output));
  }
  else
  {
    tflite::reference_ops::Mean(
        getTensorData<float>(_input), getTensorShape(_input).DimsData(), _input->shape().num_dims(),
        getTensorData<float>(_output), getTensorShape(_output).DimsData(),
        _output->shape().num_dims(), axis_data, num_axis, _params.keep_dims,
        getTensorData<int>(_temp_index.get()), getTensorData<int>(_resolved_axis.get()),
        getTensorData<float>(_temp_sum.get()));
  }
}

void Mean::evalQuantized() const
{
  const Shape &input_shape = _input->shape();
  int input_num_dims = input_shape.num_dims();
  const auto *axis_data = getTensorData<int32_t>(_axis);
  int num_axis = _axis->shape().num_elements();

  tflite::MeanParams params{};
  ResolveAxis(axis_data, num_axis, &params);

  // Defer to specialized implementation for 4D Mean across axes 1 & 2.
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
    tflite::reference_ops::Mean(
        getTensorData<uint8_t>(_input), getTensorShape(_input).DimsData(),
        _input->shape().num_dims(), getTensorData<uint8_t>(_output),
        getTensorShape(_output).DimsData(), _output->shape().num_dims(), axis_data, num_axis,
        _params.keep_dims, getTensorData<int>(_temp_index.get()),
        getTensorData<int>(_resolved_axis.get()), getTensorData<int>(_temp_sum.get()));
  }
  else
  {
    tflite::reference_ops::QuantizedMeanOrSum<>(
        getTensorData<uint8_t>(_input), _input->zero_point(), _input->scale(),
        getTensorShape(_input).DimsData(), _input->shape().num_dims(),
        getTensorData<uint8_t>(_output), _output->zero_point(), _output->scale(),
        getTensorShape(_output).DimsData(), _output->shape().num_dims(), axis_data, num_axis,
        _params.keep_dims, getTensorData<int>(_temp_index.get()),
        getTensorData<int>(_resolved_axis.get()), getTensorData<int>(_temp_sum.get()),
        /*compute_sum=*/false);
  }
}

} // namespace kernels
} // namespace luci_interpreter
