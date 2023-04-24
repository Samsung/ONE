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

#include "Builders.h"

#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/reference/integer_ops/pooling.h>
#include <tensorflow/lite/kernels/internal/reference/pooling.h>

namespace luci_interpreter
{

namespace
{

#ifndef DIS_FLOAT

void evalFloat(const circle::Tensor *input, const circle::Tensor *output,
               const circle::Pool2DOptions *options, BaseRuntimeGraph *runtime_graph)
{
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

  float activation_min{};
  float activation_max{};
  kernels::calculateActivationRange(luci_actfunc(options->fused_activation_function()),
                                    &activation_min, &activation_max);
  tflite::PoolParams params{};
  params.padding_values.height = padding_height;
  params.padding_values.width = padding_width;
  params.stride_height = options->stride_h();
  params.stride_width = options->stride_w();
  params.filter_height = options->filter_height();
  params.filter_width = options->filter_width();
  params.float_activation_min = activation_min;
  params.float_activation_max = activation_max;

  tflite::reference_ops::MaxPool(
    params, kernels::getTensorShape(input), kernels::getTensorData<float>(input_data),
    kernels::getTensorShape(output), kernels::getTensorData<float>(output_data));
}

#endif // DIS_FLOAT

#ifndef DIS_QUANT
void evalQuantized(const circle::Tensor *input, const circle::Tensor *output,
                   const circle::Pool2DOptions *options, BaseRuntimeGraph *runtime_graph)
{
  int32_t activation_min{};
  int32_t activation_max{};
  kernels::calculateActivationRangeQuantized(luci_actfunc(options->fused_activation_function()),
                                             output, &activation_min, &activation_max);

  // Compute padding
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

  tflite::PoolParams params{};
  params.padding_values.height = padding_height;
  params.padding_values.width = padding_width;
  params.stride_height = options->stride_h();
  params.stride_width = options->stride_w();
  params.filter_height = options->filter_height();
  params.filter_width = options->filter_width();
  params.quantized_activation_min = activation_min;
  params.quantized_activation_max = activation_max;

  const auto *input_data = runtime_graph->getDataByTensor(input);
  auto *output_data = runtime_graph->getDataByTensor(output);

  tflite::reference_ops::MaxPool(
    params, kernels::getTensorShape(input), kernels::getTensorData<uint8_t>(input_data),
    kernels::getTensorShape(output), kernels::getTensorData<uint8_t>(output_data));
}

void evalSInt16(const circle::Tensor *input, const circle::Tensor *output,
                const circle::Pool2DOptions *options, BaseRuntimeGraph *runtime_graph)
{
  int32_t activation_min{};
  int32_t activation_max{};
  kernels::calculateActivationRangeQuantized(luci_actfunc(options->fused_activation_function()),
                                             output, &activation_min, &activation_max);

  // Compute padding
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

  tflite::PoolParams params{};
  params.padding_values.height = padding_height;
  params.padding_values.width = padding_width;
  params.stride_height = options->stride_h();
  params.stride_width = options->stride_w();
  params.filter_height = options->filter_height();
  params.filter_width = options->filter_width();
  params.quantized_activation_min = activation_min;
  params.quantized_activation_max = activation_max;

  const auto *input_data = runtime_graph->getDataByTensor(input);
  auto *output_data = runtime_graph->getDataByTensor(output);

  tflite::reference_integer_ops::MaxPool(
    params, kernels::getTensorShape(input), kernels::getTensorData<int16_t>(input_data),
    kernels::getTensorShape(output), kernels::getTensorData<int16_t>(output_data));
}

#endif // DIS_QUANT

} // namespace

void configure_kernel_CircleMaxPool2D(const circle::Operator *cur_op,
                                      BaseRuntimeGraph *runtime_graph)
{
  const auto input_index = cur_op->inputs()->operator[](0);
  const auto output_index = cur_op->outputs()->operator[](0);

  assert(input_index != -1);
  assert(output_index != -1);

  const auto input = runtime_graph->getCircleTensorByIndex(input_index);
  const auto output = runtime_graph->getCircleTensorByIndex(output_index);

  LUCI_INTERPRETER_CHECK(Tensor::element_type(input) == Tensor::element_type(output));
  assert(Tensor::num_dims(input) == 4);

#ifndef DIS_QUANT
  if (Tensor::element_type(input) == DataType::U8)
  {
    LUCI_INTERPRETER_CHECK(std::abs(Tensor::scale(output) - Tensor::scale(input)) <= 1.0e-6);
    LUCI_INTERPRETER_CHECK(Tensor::zero_point(output) == Tensor::zero_point(input));
  }
  else if (Tensor::element_type(input) == DataType::S16)
  {
    LUCI_INTERPRETER_CHECK(std::abs(Tensor::scale(output) - Tensor::scale(input)) <= 1.0e-6);
    LUCI_INTERPRETER_CHECK(Tensor::zero_point(input) == 0 && Tensor::zero_point(output) == 0);
  }
#endif // DIS_QUANT
}

void execute_kernel_CircleMaxPool2D(const circle::Operator *cur_op, BaseRuntimeGraph *runtime_graph,
                                    bool)
{
  const auto input_index = cur_op->inputs()->operator[](0);
  const auto output_index = cur_op->outputs()->operator[](0);

  assert(input_index != -1);
  assert(output_index != -1);

  const auto input = runtime_graph->getCircleTensorByIndex(input_index);
  auto output = runtime_graph->getCircleTensorByIndex(output_index);

  const auto *options = cur_op->builtin_options_as_Pool2DOptions();

  switch (Tensor::element_type(input))
  {
#ifndef DIS_FLOAT
    case DataType::FLOAT32:
      evalFloat(input, output, options, runtime_graph);
      break;
#endif // DIS_FLOAT
#ifndef DIS_QUANT
    case DataType::U8:
      evalQuantized(input, output, options, runtime_graph);
      break;
    case DataType::S16:
      evalSInt16(input, output, options, runtime_graph);
      break;
#endif // DIS_QUANT
    default:
      assert(false && "Unsupported type.");
  }
}

} // namespace luci_interpreter
