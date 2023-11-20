/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/SpaceToBatchND.h"
#include "kernels/Utils.h"

#include "PALSpaceToBatchND.h"

#include <stdexcept>

namespace luci_interpreter
{
namespace kernels
{
namespace
{

const int kInputMinDimensionNum = 3;
const int kInputMaxDimensionNum = 4;

} // namespace

SpaceToBatchND::SpaceToBatchND(const Tensor *input, const Tensor *block_shape,
                               const Tensor *paddings, Tensor *output)
  : Kernel({input, block_shape, paddings}, {output})
{
}

void SpaceToBatchND::configure()
{
  const auto *block_shape_data = block_shape()->data<int32_t>();
  const auto *paddings_data = paddings()->data<int32_t>();
  LUCI_INTERPRETER_CHECK(input()->shape().num_dims() >= kInputMinDimensionNum);
  LUCI_INTERPRETER_CHECK(input()->shape().num_dims() <= kInputMaxDimensionNum);
  LUCI_INTERPRETER_CHECK(input()->element_type() == output()->element_type());

  int spatial_dims_num = input()->shape().num_dims() - 2;

  LUCI_INTERPRETER_CHECK(block_shape()->shape().num_dims() == 1);
  LUCI_INTERPRETER_CHECK(block_shape()->shape().dim(0) == spatial_dims_num);

  LUCI_INTERPRETER_CHECK(paddings()->shape().num_dims() == 2);
  LUCI_INTERPRETER_CHECK(paddings()->shape().dim(0) == spatial_dims_num);
  LUCI_INTERPRETER_CHECK(paddings()->shape().dim(1) == 2);

  Shape output_shape = Shape(input()->shape().num_dims());
  int output_batch_size = input()->shape().dim(0);
  for (int i = 0; i < spatial_dims_num; ++i)
  {
    int final_dim_size =
      (input()->shape().dim(i + 1) + paddings_data[i * 2] + paddings_data[i * 2 + 1]);
    LUCI_INTERPRETER_CHECK(final_dim_size % block_shape_data[i] == 0);
    output_shape.dim(i + 1) = final_dim_size / block_shape_data[i];
    output_batch_size = output_batch_size * block_shape_data[i];
  }
  output_shape.dim(0) = output_batch_size;
  output_shape.dim(input()->shape().num_dims() - 1) =
    input()->shape().dim(input()->shape().num_dims() - 1);
  output()->resize(output_shape);
}

void SpaceToBatchND::execute() const
{
  switch (input()->element_type())
  {
    tflite::SpaceToBatchParams op_params;
    case DataType::FLOAT32:
      op_params.output_offset = 0;
      luci_interpreter_pal::SpaceToBatchND(
        op_params, getTensorShape(input()), getTensorData<float>(input()),
        getTensorShape(block_shape()), getTensorData<int32_t>(block_shape()),
        getTensorShape(paddings()), getTensorData<int32_t>(paddings()), getTensorShape(output()),
        getTensorData<float>(output()));
      break;
    case DataType::U8:
      op_params.output_offset = output()->zero_point();
      luci_interpreter_pal::SpaceToBatchND(
        op_params, getTensorShape(input()), getTensorData<uint8_t>(input()),
        getTensorShape(block_shape()), getTensorData<int32_t>(block_shape()),
        getTensorShape(paddings()), getTensorData<int32_t>(paddings()), getTensorShape(output()),
        getTensorData<uint8_t>(output()));
      break;
    default:
      throw std::runtime_error("luci-intp ShapeToBatchND Unsupported type.");
  }
}

} // namespace kernels
} // namespace luci_interpreter
