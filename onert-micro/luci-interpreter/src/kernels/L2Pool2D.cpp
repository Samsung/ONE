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

#include "Builders.h"
#include "SISOKernel.h"
#include "kernels/Utils.h"
#include "PALL2Pool2D.h"

namespace luci_interpreter
{
void configure_kernel_CircleL2Pool2D(const circle::Operator *cur_op,
                                     BaseRuntimeGraph *runtime_graph)
{
  const kernels::SISOKernel siso_kernel(cur_op, runtime_graph);

  LUCI_INTERPRETER_CHECK(Tensor::element_type(siso_kernel.input()) ==
                         Tensor::element_type(siso_kernel.output()));
  LUCI_INTERPRETER_CHECK(Tensor::num_dims(siso_kernel.input()) == 4);
  LUCI_INTERPRETER_CHECK(Tensor::num_dims(siso_kernel.input()) ==
                         Tensor::num_dims(siso_kernel.output()));
}

void execute_kernel_CircleL2Pool2D(const circle::Operator *cur_op, BaseRuntimeGraph *runtime_graph)
{
  const kernels::SISOKernel siso_kernel(cur_op, runtime_graph);

  const auto *options = cur_op->builtin_options_as_Pool2DOptions();

  const auto input = siso_kernel.input();
  const auto output = siso_kernel.output();

  const int32_t input_height = Tensor::dim(input, 1);
  const int32_t input_width = Tensor::dim(input, 2);

  const int32_t output_height = kernels::computeOutputSize(
    luci_padding(options->padding()), input_height, options->filter_height(), options->stride_h());
  const int32_t output_width = kernels::computeOutputSize(
    luci_padding(options->padding()), input_width, options->filter_width(), options->stride_w());

  const auto padding_height = kernels::computePadding(options->stride_h(), 1, input_height,
                                                      options->filter_height(), output_height);
  const auto padding_width = kernels::computePadding(options->stride_w(), 1, input_width,
                                                     options->filter_width(), output_width);

  const auto *input_data = runtime_graph->getDataByTensor(input);
  auto *output_data = runtime_graph->getDataByTensor(output);

  const DataType input_type = Tensor::element_type(input);

  float activation_min{};
  float activation_max{};

#ifndef DIS_FLOAT
  kernels::calculateActivationRange(luci_actfunc(options->fused_activation_function()),
                                    &activation_min, &activation_max);
#endif // DIS_FLOAT

  luci_interpreter_pal::PoolParams params{};
  params.padding_values.height = padding_height;
  params.padding_values.width = padding_width;
  params.stride_height = options->stride_h();
  params.stride_width = options->stride_w();
  params.filter_height = options->filter_height();
  params.filter_width = options->filter_width();
  params.float_activation_min = activation_min;
  params.float_activation_max = activation_max;

  switch (input_type)
  {
#ifndef DIS_FLOAT
    case DataType::FLOAT32:
      luci_interpreter_pal::L2Pool(
        params, kernels::getTensorShape(input), kernels::getTensorData<float>(input_data),
        kernels::getTensorShape(output), kernels::getTensorData<float>(output_data));
      break;
#endif // DIS_FLOAT
    default:
      assert(false && "Unsupported type.");
  }
}

} // namespace luci_interpreter
