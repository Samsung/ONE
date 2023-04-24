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

#include "kernels/ArgMax.h"
#include "kernels/Utils.h"
#include "PALArgMax.h"

namespace luci_interpreter
{
namespace kernels
{

ArgMax::ArgMax(const Tensor *input, const Tensor *axis, Tensor *output, const ArgMaxParams &params)
  : KernelWithParams<ArgMaxParams>({input, axis}, {output}, params)
{
}

void ArgMax::configure()
{
  assert(axis()->element_type() == DataType::S32 || axis()->element_type() == DataType::S64);
  assert(input()->shape().num_dims() >= 1);
  const Shape &input_shape = input()->shape();
  const int num_dims = input_shape.num_dims();
  Shape output_shape(num_dims - 1);

  // If axis value is negative, then update by adding input_shape's num_dims.
  // If updated value also negative, then assert.
  assert(axis()->shape().num_elements() == 1);
  int axis_value = getTensorData<int32_t>(axis())[0];
  if (axis_value < 0)
    axis_value = axis_value + num_dims;
  assert(axis_value >= 0);

  int j = 0;
  for (int i = 0; i < num_dims; i++)
  {
    if (i == axis_value)
      continue;
    output_shape.dim(j++) = input_shape.dim(i);
  }

  assert(output()->element_type() == _params.output_type);

  // TODO: enable it only if kernel with dynamic shapes
  output()->resize(output_shape);
}

void ArgMax::execute() const
{

#define TF_LITE_ARG_MAX(data_type, axis_type, output_type)                                    \
  luci_interpreter_pal::ArgMinMax(getTensorShape(input()), getTensorData<data_type>(input()), \
                                  getTensorData<axis_type>(axis()), getTensorShape(output()), \
                                  getTensorData<output_type>(output()), std::greater<data_type>())
  if (axis()->element_type() == DataType::S32)
  {
    switch (_params.output_type)
    {
      case DataType::S32:
        switch (input()->element_type())
        {
          case DataType::FLOAT32:
            TF_LITE_ARG_MAX(float, int32_t, int32_t);
            break;
          case DataType::U8:
            TF_LITE_ARG_MAX(uint8_t, int32_t, int32_t);
            break;
          default:
            assert(false && "Unsupported input type.");
        }
        break;
      case DataType::S64:
        switch (input()->element_type())
        {
          case DataType::FLOAT32:
            TF_LITE_ARG_MAX(float, int32_t, int64_t);
            break;
          case DataType::U8:
            TF_LITE_ARG_MAX(uint8_t, int32_t, int64_t);
            break;
          default:
            assert(false && "Unsupported input type.");
        }
        break;
      default:
        assert(false && "Unsupported output type.");
    }
  }
  else
  {
    switch (_params.output_type)
    {
      case DataType::S32:
        switch (input()->element_type())
        {
          case DataType::FLOAT32:
            TF_LITE_ARG_MAX(float, int64_t, int32_t);
            break;
          case DataType::U8:
            TF_LITE_ARG_MAX(uint8_t, int64_t, int32_t);
            break;
          default:
            assert(false && "Unsupported input type.");
        }
        break;
      case DataType::S64:
        switch (input()->element_type())
        {
          case DataType::FLOAT32:
            TF_LITE_ARG_MAX(float, int64_t, int64_t);
            break;
          case DataType::U8:
            TF_LITE_ARG_MAX(uint8_t, int64_t, int64_t);
            break;
          default:
            assert(false && "Unsupported input type.");
        }
        break;
      default:
        assert(false && "Unsupported output type.");
    }
  }
#undef TF_LITE_ARG_MAX
}

} // namespace kernels
} // namespace luci_interpreter
