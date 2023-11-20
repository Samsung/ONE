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

#include "kernels/ResizeBilinear.h"

#include "kernels/Utils.h"

#include "PALResizeBilinear.h"

namespace luci_interpreter
{
namespace kernels
{

ResizeBilinear::ResizeBilinear(const Tensor *input, const Tensor *size, Tensor *output,
                               const ResizeBilinearParams &params)
  : KernelWithParams<ResizeBilinearParams>({input, size}, {output}, params)
{
}

void ResizeBilinear::configure()
{
  LUCI_INTERPRETER_CHECK(input()->shape().num_dims() == 4);
  LUCI_INTERPRETER_CHECK(size()->shape().num_dims() == 1);
  LUCI_INTERPRETER_CHECK(size()->element_type() == DataType::S32);
  if (params().half_pixel_centers && params().align_corners)
    throw std::runtime_error("If half_pixel_centers is True, align_corners must be False.");
  LUCI_INTERPRETER_CHECK(size()->shape().dim(0) == 2);
  Shape output_shape(4);
  output_shape.dim(0) = input()->shape().dim(0);
  output_shape.dim(1) = getTensorData<int32_t>(size())[0];
  output_shape.dim(2) = getTensorData<int32_t>(size())[1];
  output_shape.dim(3) = input()->shape().dim(3);
  output()->resize(output_shape);
}

void ResizeBilinear::execute() const
{
  tflite::ResizeBilinearParams op_params{};
  op_params.align_corners = params().align_corners;
  op_params.half_pixel_centers = params().half_pixel_centers;
  switch (output()->element_type())
  {
    case DataType::FLOAT32:
      luci_interpreter_pal::ResizeBilinear(
        op_params, getTensorShape(input()), getTensorData<float>(input()), getTensorShape(size()),
        getTensorData<int32_t>(size()), getTensorShape(output()), getTensorData<float>(output()));
      break;
    case DataType::U8:
      luci_interpreter_pal::ResizeBilinear(
        op_params, getTensorShape(input()), getTensorData<uint8_t>(input()), getTensorShape(size()),
        getTensorData<int32_t>(size()), getTensorShape(output()), getTensorData<uint8_t>(output()));
      break;
    default:
      throw std::runtime_error("luci-intp ResizeBilinear Unsupported type.");
  }
}

} // namespace kernels
} // namespace luci_interpreter
