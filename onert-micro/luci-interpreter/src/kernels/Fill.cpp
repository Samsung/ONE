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

#include "kernels/Fill.h"
#include "kernels/Utils.h"
#include "PALFill.h"

namespace luci_interpreter
{
namespace kernels
{

Fill::Fill(const Tensor *dims, const Tensor *value, Tensor *output)
  : Kernel({dims, value}, {output})
{
}

template <typename T> void Fill::configureShape()
{
  const auto dims_data = getTensorData<T>(dims());
  Shape output_shape(dims()->shape().dim(0));

  for (int i = 0; i < output_shape.num_dims(); ++i)
  {
    T data = dims_data[i];
    if (data < 0)
      assert(false && "Fill dimensions must be >= 0");

    output_shape.dim(i) = data;
  }
  // TODO: enable it only if kernel with dynamic shapes
  output()->resize(output_shape);
}

void Fill::configure()
{
  const auto dims_shape = dims()->shape();
  const auto value_shape = value()->shape();

  // Make sure the 1st input tensor is 1-D
  LUCI_INTERPRETER_CHECK(dims_shape.num_dims() == 1);

  // Make sure the 1st input tensor is int32 or int64
  LUCI_INTERPRETER_CHECK(dims()->element_type() == DataType::S32 or
                         dims()->element_type() == DataType::S64);

  // Make sure the 2nd input tensor is a scalar
  LUCI_INTERPRETER_CHECK(value_shape.num_dims() == 0)

  // Check zero point and scale for S16 and S8
  if (value()->element_type() == DataType::S16 or value()->element_type() == DataType::S8)
  {
    LUCI_INTERPRETER_CHECK(value()->scale() == output()->scale());
    LUCI_INTERPRETER_CHECK(value()->zero_point() == output()->zero_point());

    if (value()->element_type() == DataType::S16)
      LUCI_INTERPRETER_CHECK(value()->zero_point() == 0);
  }
  // Resize output
  switch (dims()->element_type())
  {
    case DataType::S32:
      configureShape<int32_t>();
      break;
    case DataType::S64:
      configureShape<int64_t>();
      break;
    default:
      assert(false && "Unsupported type.");
  }
}

void Fill::execute() const
{
  switch (output()->element_type())
  {
    case DataType::S8:
      tflite::reference_ops::Fill(getTensorShape(value()), getTensorData<int8_t>(value()),
                                  getTensorShape(output()), getTensorData<int8_t>(output()));
      break;
    case DataType::S16:
      tflite::reference_ops::Fill(getTensorShape(value()), getTensorData<int16_t>(value()),
                                  getTensorShape(output()), getTensorData<int16_t>(output()));
      break;
    case DataType::S32:
      tflite::reference_ops::Fill(getTensorShape(value()), getTensorData<int32_t>(value()),
                                  getTensorShape(output()), getTensorData<int32_t>(output()));
      break;
    case DataType::S64:
      tflite::reference_ops::Fill(getTensorShape(value()), getTensorData<int64_t>(value()),
                                  getTensorShape(output()), getTensorData<int64_t>(output()));
      break;
    case DataType::FLOAT32:
      tflite::reference_ops::Fill(getTensorShape(value()), getTensorData<float>(value()),
                                  getTensorShape(output()), getTensorData<float>(output()));
      break;
    default:
      assert(false && "Unsupported type.");
  }
}

} // namespace kernels
} // namespace luci_interpreter
