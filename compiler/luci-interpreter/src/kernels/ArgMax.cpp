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

#include "kernels/ArgMax.h"
#include "kernels/Utils.h"
#include <tensorflow/lite/kernels/internal/optimized/optimized_ops.h>

namespace luci_interpreter
{
namespace kernels
{

ArgMax::ArgMax(const Tensor *input, const Tensor *axis, Tensor *output, const ArgMaxParams &params)
    : KernelWithParams<ArgMaxParams>(params), _input(input), _axis(axis), _output(output)
{
}

void ArgMax::configure()
{
  assert(_axis->element_type() == DataType::S32 || _axis->element_type() == DataType::S64);
  assert(_input->shape().num_dims() >= 1);
  const Shape &input_shape = _input->shape();
  const int num_dims = input_shape.num_dims();
  Shape output_shape(num_dims - 1);

  // If axis value is negative, then update by adding input_shape's num_dims.
  // If updated value also negative, then assert.
  assert(_axis->shape().num_elements() == 1);
  int axis = getTensorData<int32_t>(_axis)[0];
  if (axis < 0)
    axis = axis + num_dims;
  assert(axis >= 0);

  int j = 0;
  for (int i = 0; i < num_dims; i++)
  {
    if (i == axis)
      continue;
    output_shape.dim(j++) = input_shape.dim(i);
  }

  assert(_output->element_type() == _params.output_type);

  _output->resize(output_shape);
}

void ArgMax::execute() const
{

#define TF_LITE_ARG_MAX(data_type, axis_type, output_type)                                   \
  tflite::optimized_ops::ArgMinMax(getTensorShape(_input), getTensorData<data_type>(_input), \
                                   getTensorData<axis_type>(_axis), getTensorShape(_output), \
                                   getTensorData<output_type>(_output), std::greater<data_type>())
  if (_axis->element_type() == DataType::S32)
  {
    switch (_params.output_type)
    {
      case DataType::S32:
        switch (_input->element_type())
        {
          case DataType::FLOAT32:
            TF_LITE_ARG_MAX(float, int32_t, int32_t);
            break;
          case DataType::U8:
            TF_LITE_ARG_MAX(uint8_t, int32_t, int32_t);
            break;
          default:
            throw std::runtime_error("Unsupported input type.");
        }
        break;
      case DataType::S64:
        switch (_input->element_type())
        {
          case DataType::FLOAT32:
            TF_LITE_ARG_MAX(float, int32_t, int64_t);
            break;
          case DataType::U8:
            TF_LITE_ARG_MAX(uint8_t, int32_t, int64_t);
            break;
          default:
            throw std::runtime_error("Unsupported input type.");
        }
        break;
      default:
        throw std::runtime_error("Unsupported output type.");
    }
  }
  else
  {
    switch (_params.output_type)
    {
      case DataType::S32:
        switch (_input->element_type())
        {
          case DataType::FLOAT32:
            TF_LITE_ARG_MAX(float, int64_t, int32_t);
            break;
          case DataType::U8:
            TF_LITE_ARG_MAX(uint8_t, int64_t, int32_t);
            break;
          default:
            throw std::runtime_error("Unsupported input type.");
        }
        break;
      case DataType::S64:
        switch (_input->element_type())
        {
          case DataType::FLOAT32:
            TF_LITE_ARG_MAX(float, int64_t, int64_t);
            break;
          case DataType::U8:
            TF_LITE_ARG_MAX(uint8_t, int64_t, int64_t);
            break;
          default:
            throw std::runtime_error("Unsupported input type.");
        }
        break;
      default:
        throw std::runtime_error("Unsupported output type.");
    }
  }
#undef TF_LITE_ARG_MAX
}

} // namespace kernels
} // namespace luci_interpreter
