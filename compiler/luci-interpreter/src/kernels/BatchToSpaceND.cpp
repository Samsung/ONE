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

#include "kernels/BatchToSpaceND.h"
#include "kernels/Utils.h"

#include "PALBatchToSpaceND.h"

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

BatchToSpaceND::BatchToSpaceND(const Tensor *input, const Tensor *block_shape, const Tensor *crops,
                               Tensor *output)
  : Kernel({input, block_shape, crops}, {output})
{
}

void BatchToSpaceND::configure()
{

  const auto *block_shape_data = block_shape()->data<int32_t>();
  const auto *crops_data = crops()->data<int32_t>();
  LUCI_INTERPRETER_CHECK(input()->shape().num_dims() >= kInputMinDimensionNum);
  LUCI_INTERPRETER_CHECK(input()->shape().num_dims() <= kInputMaxDimensionNum);
  LUCI_INTERPRETER_CHECK(input()->element_type() == output()->element_type());

  int spatial_dims_num = input()->shape().num_dims() - 2;

  LUCI_INTERPRETER_CHECK(block_shape()->shape().num_dims() == 1);
  LUCI_INTERPRETER_CHECK(block_shape()->shape().dim(0) == spatial_dims_num);

  LUCI_INTERPRETER_CHECK(crops()->shape().num_dims() == 2);
  LUCI_INTERPRETER_CHECK(crops()->shape().dim(0) == spatial_dims_num);
  LUCI_INTERPRETER_CHECK(crops()->shape().dim(1) == 2);
  for (int i = 0; i < spatial_dims_num * 2; ++i)
  {
    LUCI_INTERPRETER_CHECK(crops_data[i] >= 0);
  }

  Shape output_shape = Shape(input()->shape().num_dims());
  int output_batch_size = input()->shape().dim(0);
  for (int i = 0; i < spatial_dims_num; ++i)
  {
    LUCI_INTERPRETER_CHECK(output_batch_size % block_shape_data[i] == 0);
    output_batch_size = output_batch_size / block_shape_data[i];
    output_shape.dim(i + 1) =
      input()->shape().dim(i + 1) * block_shape_data[i] - crops_data[i * 2] - crops_data[i * 2 + 1];
  }

  output_shape.dim(0) = output_batch_size;
  output_shape.dim(input()->shape().num_dims() - 1) =
    input()->shape().dim(input()->shape().num_dims() - 1);
  output()->resize(output_shape);
}

void BatchToSpaceND::execute() const
{
  switch (input()->element_type())
  {
    case DataType::FLOAT32:
      luci_interpreter_pal::BatchToSpaceND(
        getTensorShape(input()), getTensorData<float>(input()), getTensorShape(block_shape()),
        getTensorData<int32_t>(block_shape()), getTensorShape(crops()),
        getTensorData<int32_t>(crops()), getTensorShape(output()), getTensorData<float>(output()));
      break;
    case DataType::U8:
      luci_interpreter_pal::BatchToSpaceND(
        getTensorShape(input()), getTensorData<uint8_t>(input()), getTensorShape(block_shape()),
        getTensorData<int32_t>(block_shape()), getTensorShape(crops()),
        getTensorData<int32_t>(crops()), getTensorShape(output()),
        getTensorData<uint8_t>(output()));
      break;
    default:
      throw std::runtime_error("luci-intp BatchToSpaceND Unsupported type.");
  }
}

} // namespace kernels
} // namespace luci_interpreter
