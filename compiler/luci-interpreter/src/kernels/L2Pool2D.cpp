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

#include "kernels/L2Pool2D.h"

#include "kernels/Utils.h"

#include "PALL2Pool2D.h"

#include <stdexcept>

namespace luci_interpreter
{

namespace kernels
{

L2Pool2D::L2Pool2D(const Tensor *input, Tensor *output, const Pool2DParams &params)
  : KernelWithParams<Pool2DParams>({input}, {output}, params)
{
}

void L2Pool2D::configure()
{
  LUCI_INTERPRETER_CHECK(input()->shape().num_dims() == 4);
  LUCI_INTERPRETER_CHECK(input()->element_type() == output()->element_type());

  int batches = input()->shape().dim(0);
  int height = input()->shape().dim(1);
  int width = input()->shape().dim(2);
  int channels_out = input()->shape().dim(3);

  // Matching GetWindowedOutputSize in TensorFlow.
  auto padding = params().padding;
  int out_width, out_height;
  out_width = computeOutputSize(padding, width, params().filter_width, params().stride_width, 1);
  out_height =
    computeOutputSize(padding, height, params().filter_height, params().stride_height, 1);
  _padding_width =
    computePadding(params().stride_width, 1, width, params().filter_width, out_width);
  _padding_height =
    computePadding(params().stride_height, 1, height, params().filter_height, out_height);

  LUCI_INTERPRETER_CHECK(input()->element_type() == DataType::FLOAT32);
  output()->resize({batches, out_height, out_width, channels_out});
}

void L2Pool2D::execute() const
{
  switch (input()->element_type())
  {
    case DataType::FLOAT32:
      float activation_min, activation_max;
      calculateActivationRange(params().activation, &activation_min, &activation_max);
      tflite::PoolParams op_params;
      op_params.stride_height = params().stride_height;
      op_params.stride_width = params().stride_width;
      op_params.filter_height = params().filter_height;
      op_params.filter_width = params().filter_width;
      op_params.padding_values.height = _padding_height;
      op_params.padding_values.width = _padding_width;
      op_params.float_activation_min = activation_min;
      op_params.float_activation_max = activation_max;
      luci_interpreter_pal::L2Pool(op_params, getTensorShape(input()),
                                   getTensorData<float>(input()), getTensorShape(output()),
                                   getTensorData<float>(output()));
      break;
    default:
      throw std::runtime_error("luci-intp L2Pool2D Unsupported type.");
  }
}

} // namespace kernels
} // namespace luci_interpreter
