/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

#include "kernels/StridedSlice.h"

#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/reference/strided_slice.h>

#include <stdexcept>

namespace luci_interpreter
{

namespace kernels
{

StridedSlice::StridedSlice(const Tensor *input, const Tensor *begin, const Tensor *end,
                           const Tensor *strides, Tensor *output, const StridedSliceParams &params)
  : KernelWithParams<StridedSliceParams>({input, begin, end, strides}, {output}, params)
{
}

void StridedSlice::configure()
{
  assert(begin()->shape().num_dims() == 1);
  assert(end()->shape().num_dims() == 1);
  assert(strides()->shape().num_dims() == 1);
  assert(input()->element_type() == output()->element_type());
  assert(begin()->element_type() == DataType::S32);
  assert(end()->element_type() == DataType::S32);
  assert(strides()->element_type() == DataType::S32);
  assert(input()->shape().num_dims() <= 4);
  if (params().ellipsis_mask != 0)
  {
    throw std::runtime_error("ellipsis_mask is not implemented yet.");
  }
  if (params().new_axis_mask != 0)
  {
    throw std::runtime_error("new_axis_mask is not implemented yet.");
  }
  if (input()->element_type() == DataType::U8)
  {
    assert(input()->scale() == output()->scale());
    assert(input()->zero_point() == output()->zero_point());
  }
  tflite::StridedSliceParams op_params{};
  op_params.start_indices_count = input()->shape().num_dims();
  op_params.stop_indices_count = input()->shape().num_dims();
  op_params.strides_count = input()->shape().num_dims();

  for (int i = 0; i < input()->shape().num_dims(); i++)
  {
    op_params.start_indices[i] = getTensorData<int32_t>(begin())[i];
    op_params.stop_indices[i] = getTensorData<int32_t>(end())[i];
    op_params.strides[i] = getTensorData<int32_t>(strides())[i];
  }
  op_params.begin_mask = params().begin_mask;
  op_params.ellipsis_mask = 0;
  op_params.end_mask = params().end_mask;
  op_params.new_axis_mask = 0;
  op_params.shrink_axis_mask = params().shrink_axis_mask;
  std::vector<int32_t> output_shape_vector;
  for (int i = 0; i < input()->shape().num_dims(); i++)
  {
    int idx = input()->shape().num_dims() - i - 1;
    int32_t stride = getTensorData<int32_t>(strides())[idx];
    assert(stride != 0);
    int32_t begin = ::tflite::strided_slice::StartForAxis(op_params, getTensorShape(input()), idx);
    int32_t end =
      ::tflite::strided_slice::StopForAxis(op_params, getTensorShape(input()), idx, begin);

    const bool shrink_axis = params().shrink_axis_mask & (1 << idx);
    if (shrink_axis)
    {
      end = begin + 1;
    }

    int32_t dim_shape = std::ceil((end - begin) / static_cast<float>(stride));
    dim_shape = dim_shape < 0 ? 0 : dim_shape;
    if (!shrink_axis)
    {
      output_shape_vector.push_back(dim_shape);
    }
  }
  Shape output_shape = Shape(output_shape_vector.size());
  for (size_t i = 0; i < output_shape_vector.size(); i++)
  {
    output_shape.dim(i) = output_shape_vector[output_shape_vector.size() - i - 1];
  }
  output()->resize(output_shape);
}

void StridedSlice::execute() const
{
  tflite::StridedSliceParams op_params{};
  op_params.start_indices_count = input()->shape().num_dims();
  op_params.stop_indices_count = input()->shape().num_dims();
  op_params.strides_count = input()->shape().num_dims();

  for (int i = 0; i < input()->shape().num_dims(); i++)
  {
    op_params.start_indices[i] = getTensorData<int32_t>(begin())[i];
    op_params.stop_indices[i] = getTensorData<int32_t>(end())[i];
    op_params.strides[i] = getTensorData<int32_t>(strides())[i];
  }
  op_params.begin_mask = params().begin_mask;
  op_params.ellipsis_mask = 0;
  op_params.end_mask = params().end_mask;
  op_params.new_axis_mask = 0;
  op_params.shrink_axis_mask = params().shrink_axis_mask;

  switch (input()->element_type())
  {
    case DataType::FLOAT32:
      tflite::reference_ops::StridedSlice(op_params, getTensorShape(input()),
                                          getTensorData<float>(input()), getTensorShape(output()),
                                          getTensorData<float>(output()));
      break;
    case DataType::U8:
      tflite::reference_ops::StridedSlice(op_params, getTensorShape(input()),
                                          getTensorData<uint8_t>(input()), getTensorShape(output()),
                                          getTensorData<uint8_t>(output()));
      break;
    case DataType::S32:
      tflite::reference_ops::StridedSlice(op_params, getTensorShape(input()),
                                          getTensorData<int32_t>(input()), getTensorShape(output()),
                                          getTensorData<int32_t>(output()));
      break;
    default:
      throw std::runtime_error("luci-intp StridedSlice Unsupported type.");
  }
}

} // namespace kernels
} // namespace luci_interpreter
