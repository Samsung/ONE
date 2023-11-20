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

#include <tensorflow/lite/kernels/internal/reference/reduce.h>

#include <stdexcept>

namespace luci_interpreter
{
namespace kernels
{

static void resolveAxes(const int32_t *axes_data, int num_axes, tflite::MeanParams *params)
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
static int getAxisReductionCount(const int32_t *axes_data, int num_axes, int input_num_dims)
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

static Shape getOutputShape(const Shape &input_shape, const int32_t *axes_data, int num_axes,
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

Mean::Mean(const Tensor *input, const Tensor *axes, Tensor *output, Tensor *temp_index,
           Tensor *resolved_axes, Tensor *temp_sum, const ReducerParams &params)
  : KernelWithParams<ReducerParams>({input, axes}, {output, temp_index, resolved_axes, temp_sum},
                                    params)
{
}

void Mean::configure()
{
  LUCI_INTERPRETER_CHECK(input()->element_type() == output()->element_type());
  LUCI_INTERPRETER_CHECK(axes()->element_type() == DataType::S32);
  if (input()->element_type() == DataType::S16)
  {
    LUCI_INTERPRETER_CHECK(input()->zero_point() == 0 && output()->zero_point() == 0);
  }

  const Shape &input_shape = input()->shape();
  int input_num_dims = input_shape.num_dims();

  const auto *axes_data = getTensorData<int32_t>(axes());
  int num_axes = axes()->shape().num_elements();
  assert(num_axes <= 4);

  Shape output_shape = getOutputShape(input_shape, axes_data, num_axes, _params.keep_dims);
  output()->resize(output_shape);

  tflite::MeanParams params{};
  resolveAxes(axes_data, num_axes, &params);
  _need_temporaries = !(
    _params.keep_dims && input_num_dims == 4 && params.axis_count == 2 &&
    ((params.axis[0] == 1 && params.axis[1] == 2) || (params.axis[0] == 2 && params.axis[1] == 1)));
  if (_need_temporaries)
  {
    auto temp_index = getOutputTensors()[1];
    auto resolved_axes = getOutputTensors()[2];
    auto temp_sum = getOutputTensors()[3];

    temp_index->resize(Shape(input_num_dims));
    resolved_axes->resize(Shape(num_axes));
    temp_sum->resize(output()->shape());
  }
  else
  {
    auto temp_index = getOutputTensors()[1];
    auto resolved_axes = getOutputTensors()[2];
    auto temp_sum = getOutputTensors()[3];

    temp_index->set_allocatable(false);
    resolved_axes->set_allocatable(false);
    temp_sum->set_allocatable(false);
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
    case DataType::S16:
      evalQuantizedS16();
      break;
    default:
      throw std::runtime_error("luci-intp Mean Unsupported type.");
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

  auto temp_index = getOutputTensors()[1];
  auto resolved_axes = getOutputTensors()[2];
  auto temp_sum = getOutputTensors()[3];

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
    tflite::reference_ops::Mean(getTensorData<float>(input()), getTensorShape(input()).DimsData(),
                                input()->shape().num_dims(), getTensorData<float>(output()),
                                getTensorShape(output()).DimsData(), output()->shape().num_dims(),
                                axes_data, num_axes, _params.keep_dims,
                                getTensorData<int>(temp_index), getTensorData<int>(resolved_axes),
                                getTensorData<float>(temp_sum));
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

  auto temp_index = getOutputTensors()[1];
  auto resolved_axes = getOutputTensors()[2];
  auto temp_sum = getOutputTensors()[3];

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
    tflite::reference_ops::Mean(getTensorData<uint8_t>(input()), getTensorShape(input()).DimsData(),
                                input()->shape().num_dims(), getTensorData<uint8_t>(output()),
                                getTensorShape(output()).DimsData(), output()->shape().num_dims(),
                                axes_data, num_axes, _params.keep_dims,
                                getTensorData<int>(temp_index), getTensorData<int>(resolved_axes),
                                getTensorData<int>(temp_sum));
  }
  else
  {
    tflite::reference_ops::QuantizedMeanOrSum<>(
      getTensorData<uint8_t>(input()), input()->zero_point(), input()->scale(),
      getTensorShape(input()).DimsData(), input()->shape().num_dims(),
      getTensorData<uint8_t>(output()), output()->zero_point(), output()->scale(),
      getTensorShape(output()).DimsData(), output()->shape().num_dims(), axes_data, num_axes,
      _params.keep_dims, getTensorData<int>(temp_index), getTensorData<int>(resolved_axes),
      getTensorData<int>(temp_sum),
      /*compute_sum=*/false);
  }
}

void Mean::evalQuantizedS16() const
{
  const auto *input_data = getTensorData<int16_t>(input());
  auto *output_data = getTensorData<int16_t>(output());

  const Shape &input_shape = input()->shape();
  const Shape &output_shape = output()->shape();

  const auto *axes_data = getTensorData<int32_t>(axes());
  const int num_axes = axes()->shape().num_elements();

  constexpr int32_t output_min = -std::numeric_limits<int16_t>::max();
  constexpr int32_t output_max = std::numeric_limits<int16_t>::max();

  // Defer to specialized implementation for 4D Mean across axes 1 & 2.
  if (_params.keep_dims && input_shape.num_dims() == 4 && num_axes == 2 &&
      ((axes_data[0] == 1 && axes_data[1] == 2) || (axes_data[0] == 2 && axes_data[1] == 1)))
  {
    const int32_t batches = input_shape.dim(0);
    const int32_t input_height = input_shape.dim(1);
    const int32_t input_width = input_shape.dim(2);
    const int32_t depth = input_shape.dim(3);
    assert(output_shape.num_dims() == 4);
    assert(output_shape.dim(0) == batches);
    assert(output_shape.dim(1) == 1);
    assert(output_shape.dim(2) == 1);
    assert(output_shape.dim(3) == depth);

    const double real_multiplier =
      static_cast<double>(input()->scale()) / static_cast<double>(output()->scale());

    int32_t output_multiplier{};
    int output_shift{};
    quantizeMultiplier(real_multiplier, &output_multiplier, &output_shift);

    const int32_t num_elements_in_axes = input_height * input_width;

    for (int32_t batch = 0; batch < batches; ++batch)
    {
      for (int32_t c = 0; c < depth; ++c)
      {
        int32_t acc = 0;
        for (int32_t in_y = 0; in_y < input_height; ++in_y)
        {
          for (int32_t in_x = 0; in_x < input_width; ++in_x)
          {
            acc += input_data[calcOffset(input_shape, batch, in_y, in_x, c)];
          }
        }
        int32_t scaled_acc =
          tflite::MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
        // Divide by the number of elements rounding to the nearest integer.
        scaled_acc = scaled_acc > 0
                       ? (scaled_acc + num_elements_in_axes / 2) / num_elements_in_axes
                       : (scaled_acc - num_elements_in_axes / 2) / num_elements_in_axes;

        scaled_acc = std::max(scaled_acc, output_min);
        scaled_acc = std::min(scaled_acc, output_max);

        output_data[calcOffset(output_shape, batch, 0, 0, c)] = scaled_acc;
      }
    }
  }
  else
  {
    throw std::runtime_error("Unsupported configuration.");
  }
}

} // namespace kernels
} // namespace luci_interpreter
