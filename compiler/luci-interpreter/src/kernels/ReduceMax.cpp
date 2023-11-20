/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/ReduceMax.h"

#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/reference/reduce.h>

#include <stdexcept>
#include <limits>

namespace luci_interpreter
{
namespace kernels
{

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

ReduceMax::ReduceMax(const Tensor *input, const Tensor *axes, Tensor *output, Tensor *temp_index,
                     Tensor *resolved_axes, const ReducerParams &params)
  : KernelWithParams<ReducerParams>({input, axes}, {output, temp_index, resolved_axes}, params)
{
}

void ReduceMax::configure()
{
  LUCI_INTERPRETER_CHECK(input()->element_type() == output()->element_type());
  LUCI_INTERPRETER_CHECK(axes()->element_type() == DataType::S32);

  const Shape &input_shape = input()->shape();
  int input_num_dims = input_shape.num_dims();

  const auto *axes_data = getTensorData<int32_t>(axes());
  int num_axes = axes()->shape().num_elements();
  LUCI_INTERPRETER_CHECK(num_axes <= 4);

  // We compute shapes of outputs in configure, assuming that outputs have
  // static shape
  // TODO Support dynamic shape
  Shape output_shape = getOutputShape(input_shape, axes_data, num_axes, _params.keep_dims);
  output()->resize(output_shape);

  auto temp_index = getOutputTensors()[1];
  auto resolved_axes = getOutputTensors()[2];

  temp_index->resize(Shape(input_num_dims));
  resolved_axes->resize(Shape(num_axes));
}

void ReduceMax::execute() const
{
  switch (input()->element_type())
  {
    case DataType::FLOAT32:
      evalFloat();
      break;
    // TODO Support quantized kernels
    default:
      throw std::runtime_error("luci-intp ReduceMax Unsupported type.");
  }
}

void ReduceMax::evalFloat() const
{
  const auto *axes_data = getTensorData<int32_t>(axes());
  int num_axes = axes()->shape().num_elements();

  auto temp_index = getOutputTensors()[1];
  auto resolved_axes = getOutputTensors()[2];

  int num_resolved_axis = 0;
  LUCI_INTERPRETER_CHECK(
    tflite::reference_ops::ResolveAxis(input()->shape().num_dims(), axes_data, num_axes,
                                       getTensorData<int>(resolved_axes), &num_resolved_axis));

  float init_value = std::numeric_limits<float>::lowest();
  tflite::reference_ops::ReduceGeneric<float>(
    getTensorData<float>(input()), getTensorShape(input()).DimsData(), input()->shape().num_dims(),
    getTensorData<float>(output()), getTensorShape(output()).DimsData(),
    output()->shape().num_dims(), axes_data, num_axes, _params.keep_dims,
    getTensorData<int>(temp_index), getTensorData<int>(resolved_axes), init_value,
    [](const float current, const float in) -> float { return (in > current) ? in : current; });
}

} // namespace kernels
} // namespace luci_interpreter
