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

static void resolveAxes(const int *axes_data, int num_axes, tflite::MeanParams *params)
{
  params->axis_count = num_axes;
  for (int i = 0; i < num_axes; ++i)
  {
    params->axis[i] = static_cast<int16>(axes_data[i]);
  }
  for (int i = num_axes; i < 4; ++i)
  {
    params->axis[i] = 1;
  }
}

// Returns the number of axes that will be reduced. Removes duplicates.
static int getAxisReductionCount(const int *axes_data, int num_axes, int input_num_dims)
{
  int reduction_count = num_axes;
  for (int i = 0; i < num_axes; ++i)
  {
    int current = axes_data[i] >= 0 ? axes_data[i] : axes_data[i] + input_num_dims;
    assert(current >= 0 && current < input_num_dims);
    for (int j = 0; j < i; j++)
    {
      int previous = axes_data[j] >= 0 ? axes_data[j] : axes_data[j] + input_num_dims;
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

static Shape getOutputShape(const Shape &input_shape, const int *axes_data, int num_axes,
                            bool keep_dims)
{
  int input_num_dims = input_shape.num_dims();
  if (input_num_dims == 0)
  {
    return Shape(0);
  }

  if (keep_dims)
  {
    Shape output_shape(input_num_dims);
    for (int idx = 0; idx < input_num_dims; ++idx)
    {
      bool is_axis = false;
      for (int axis_idx = 0; axis_idx < num_axes; ++axis_idx)
      {
        if (axes_data[axis_idx] == idx || axes_data[axis_idx] + input_num_dims == idx)
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
    return output_shape;
  }
  else
  {
    int num_reduce_axes = getAxisReductionCount(axes_data, num_axes, input_num_dims);
    Shape output_shape(input_num_dims - num_reduce_axes);
    int num_skip_axes = 0;
    for (int idx = 0; idx < input_num_dims; ++idx)
    {
      bool is_axis = false;
      for (int axis_idx = 0; axis_idx < num_axes; ++axis_idx)
      {
        if (axes_data[axis_idx] == idx || axes_data[axis_idx] + input_num_dims == idx)
        {
          ++num_skip_axes;
          is_axis = true;
          break;
        }
      }
      if (!is_axis)
      {
        output_shape.dim(idx - num_skip_axes) = input_shape.dim(idx);
      }
    }
    return output_shape;
  }
}

Mean::Mean(const Tensor *input, const Tensor *axes, Tensor *output, const ReducerParams &params)
    : KernelWithParams<ReducerParams>({input, axes}, {output}, params)
{
}

void Mean::configure()
{
  assert(input()->element_type() == output()->element_type());
  assert(axes()->element_type() == DataType::S32);
  const Shape &input_shape = input()->shape();
  int input_num_dims = input_shape.num_dims();

  const auto *axes_data = getTensorData<int32_t>(axes());
  int num_axes = axes()->shape().num_elements();
  assert(num_axes <= 4);

  Shape output_shape = getOutputShape(input_shape, axes_data, num_axes, _params.keep_dims);
  output()->resize(output_shape);

  tflite::MeanParams params{};
  resolveAxes(axes_data, num_axes, &params);
  const bool need_temporaries =
      !(_params.keep_dims && input_num_dims == 4 && params.axis_count == 2 &&
        ((params.axis[0] == 1 && params.axis[1] == 2) ||
         (params.axis[0] == 2 && params.axis[1] == 1)));
  if (need_temporaries)
  {
    _temp_index =
        std::make_unique<Tensor>(DataType::S32, Shape(input_num_dims), AffineQuantization{}, "");
    _resolved_axes =
        std::make_unique<Tensor>(DataType::S32, Shape(num_axes), AffineQuantization{}, "");
    _temp_sum = std::make_unique<Tensor>(input()->element_type(), output()->shape(),
                                         AffineQuantization{}, "");
  }
}

void Mean::execute() const
{
  switch (input()->element_type())
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
  const Shape &input_shape = input()->shape();
  int input_num_dims = input_shape.num_dims();
  const auto *axes_data = getTensorData<int32_t>(axes());
  int num_axes = axes()->shape().num_elements();

  tflite::MeanParams params{};
  resolveAxes(axes_data, num_axes, &params);

  // Defer to specialized implementation for 4D Mean across axes 1 & 2.
  if (_params.keep_dims && input_num_dims == 4 && params.axis_count == 2 &&
      ((params.axis[0] == 1 && params.axis[1] == 2) ||
       (params.axis[0] == 2 && params.axis[1] == 1)))
  {
    tflite::reference_ops::Mean(params, getTensorShape(input()), getTensorData<float>(input()),
                                getTensorShape(output()), getTensorData<float>(output()));
  }
  else
  {
    tflite::reference_ops::Mean(
        getTensorData<float>(input()), getTensorShape(input()).DimsData(),
        input()->shape().num_dims(), getTensorData<float>(output()),
        getTensorShape(output()).DimsData(), output()->shape().num_dims(), axes_data, num_axes,
        _params.keep_dims, getTensorData<int>(_temp_index.get()),
        getTensorData<int>(_resolved_axes.get()), getTensorData<float>(_temp_sum.get()));
  }
}

void Mean::evalQuantized() const
{
  const Shape &input_shape = input()->shape();
  int input_num_dims = input_shape.num_dims();
  const auto *axes_data = getTensorData<int32_t>(axes());
  int num_axes = axes()->shape().num_elements();

  tflite::MeanParams params{};
  resolveAxes(axes_data, num_axes, &params);

  // Defer to specialized implementation for 4D Mean across axes 1 & 2.
  if (_params.keep_dims && input_num_dims == 4 && params.axis_count == 2 &&
      ((params.axis[0] == 1 && params.axis[1] == 2) ||
       (params.axis[0] == 2 && params.axis[1] == 1)))
  {
    tflite::reference_ops::Mean(params, getTensorShape(input()), getTensorData<uint8_t>(input()),
                                input()->zero_point(), input()->scale(), getTensorShape(output()),
                                getTensorData<uint8_t>(output()), output()->zero_point(),
                                output()->scale());
  }
  else if (input()->zero_point() == output()->zero_point() && input()->scale() == output()->scale())
  {
    tflite::reference_ops::Mean(
        getTensorData<uint8_t>(input()), getTensorShape(input()).DimsData(),
        input()->shape().num_dims(), getTensorData<uint8_t>(output()),
        getTensorShape(output()).DimsData(), output()->shape().num_dims(), axes_data, num_axes,
        _params.keep_dims, getTensorData<int>(_temp_index.get()),
        getTensorData<int>(_resolved_axes.get()), getTensorData<int>(_temp_sum.get()));
  }
  else
  {
    tflite::reference_ops::QuantizedMeanOrSum<>(
        getTensorData<uint8_t>(input()), input()->zero_point(), input()->scale(),
        getTensorShape(input()).DimsData(), input()->shape().num_dims(),
        getTensorData<uint8_t>(output()), output()->zero_point(), output()->scale(),
        getTensorShape(output()).DimsData(), output()->shape().num_dims(), axes_data, num_axes,
        _params.keep_dims, getTensorData<int>(_temp_index.get()),
        getTensorData<int>(_resolved_axes.get()), getTensorData<int>(_temp_sum.get()),
        /*compute_sum=*/false);
  }
}

} // namespace kernels
} // namespace luci_interpreter
