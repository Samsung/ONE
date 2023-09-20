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

#include "PALDepthwiseConv2D.h"

namespace luci_interpreter
{

namespace
{

// TODO: reduce code duplication with Conv2D
int32_t compute_padding_h(const circle::Tensor *input, const circle::Tensor *filter,
                          const circle::DepthwiseConv2DOptions *options)
{
  const int32_t input_height = Tensor::dim(input, 1);
  const int32_t filter_height = Tensor::dim(filter, 1);
  const int32_t output_height =
    kernels::computeOutputSize(luci_padding(options->padding()), input_height, filter_height,
                               options->stride_h(), options->dilation_h_factor());

  const auto padding_height = kernels::computePadding(
    options->stride_h(), options->dilation_h_factor(), input_height, filter_height, output_height);
  return padding_height;
}

int32_t compute_padding_w(const circle::Tensor *input, const circle::Tensor *filter,
                          const circle::DepthwiseConv2DOptions *options)
{
  const int32_t input_width = Tensor::dim(input, 2);
  const int32_t filter_width = Tensor::dim(filter, 2);
  const int32_t output_width =
    kernels::computeOutputSize(luci_padding(options->padding()), input_width, filter_width,
                               options->stride_w(), options->dilation_w_factor());

  const auto padding_width = kernels::computePadding(
    options->stride_w(), options->dilation_w_factor(), input_width, filter_width, output_width);

  return padding_width;
}

#ifndef DIS_FLOAT

void evalFloat(const circle::Tensor *input, const circle::Tensor *filter,
               const circle::Tensor *bias, const circle::Tensor *output,
               const circle::DepthwiseConv2DOptions *options, BaseRuntimeGraph *runtime_graph)
{
  float activation_min{};
  float activation_max{};
  kernels::calculateActivationRange(luci_actfunc(options->fused_activation_function()),
                                    &activation_min, &activation_max);

  luci_interpreter_pal::ConvParams params{};
  params.padding_values.height = compute_padding_h(input, filter, options);
  params.padding_values.width = compute_padding_w(input, filter, options);
  params.stride_height = options->stride_h();
  params.stride_width = options->stride_w();
  params.dilation_height_factor = options->dilation_h_factor();
  params.dilation_width_factor = options->dilation_w_factor();
  params.float_activation_min = activation_min;
  params.float_activation_max = activation_max;
  params.depth_multiplier = options->depth_multiplier();

  auto *input_data = runtime_graph->getDataByTensor(input);
  auto *output_data = runtime_graph->getDataByTensor(output);

  auto *filter_data = runtime_graph->getConstDataByTensor(filter);
  auto *bias_data = runtime_graph->getConstDataByTensor(bias);

  int32_t input_shape[kMaxSmallSize];
  kernels::getTensorDims(input, runtime_graph, input_shape);

  int32_t filter_shape[kMaxSmallSize];
  kernels::getTensorDims(filter, runtime_graph, filter_shape);

  int32_t output_shape[kMaxSmallSize];
  kernels::getTensorDims(output, runtime_graph, output_shape);

  luci_interpreter_pal::DepthwiseConv2D(
    params, input_shape, kernels::getTensorData<float>(input_data), filter_shape,
    kernels::getTensorData<float>(filter_data), kernels::getTensorData<float>(bias_data),
    output_shape, kernels::getTensorData<float>(output_data));
}

#endif // DIS_FLOAT

} // namespace

void configure_kernel_CircleDepthwiseConv2D(const circle::Operator *cur_op,
                                            BaseRuntimeGraph *runtime_graph)
{
  const auto input_index = cur_op->inputs()->operator[](0);
  const auto filter_index = cur_op->inputs()->operator[](1);
  const auto bias_index = cur_op->inputs()->operator[](2);
  const auto output_index = cur_op->outputs()->operator[](0);

  assert(input_index != -1);
  assert(filter_index != -1);
  assert(output_index != -1);

  const auto input = runtime_graph->getCircleTensorByIndex(input_index);
  const auto filter = runtime_graph->getCircleTensorByIndex(filter_index);
  const auto bias = runtime_graph->getCircleTensorByIndex(bias_index);
  const auto output = runtime_graph->getCircleTensorByIndex(output_index);

  assert(input != nullptr);
  assert(filter != nullptr);

  auto filter_data = runtime_graph->getConstDataByTensor(filter);

  assert(filter_data != nullptr);

  const auto *options = cur_op->builtin_options_as_DepthwiseConv2DOptions();

  if (Tensor::element_type(input) == DataType::FLOAT32 &&
      Tensor::element_type(filter) == DataType::FLOAT32)
  {
    LUCI_INTERPRETER_CHECK(bias == nullptr || Tensor::element_type(bias) == DataType::FLOAT32);
  }
  else
  {
    assert(false && "Unsupported type.");
  }
  LUCI_INTERPRETER_CHECK(Tensor::element_type(output) == Tensor::element_type(input));
  LUCI_INTERPRETER_CHECK(Tensor::num_dims(input) == 4 && Tensor::num_dims(filter) == 4);

  const int32_t output_depth = Tensor::dim(output, 3);
  LUCI_INTERPRETER_CHECK(bias == nullptr ||
                         (Tensor::num_dims(bias) == 1 && Tensor::dim(bias, 0) == output_depth));

  switch (options->fused_activation_function())
  {
    case circle::ActivationFunctionType_NONE:
    case circle::ActivationFunctionType_RELU:
    case circle::ActivationFunctionType_RELU6:
    case circle::ActivationFunctionType_RELU_N1_TO_1:
      break;
    default:
      assert(false && "Unsupported fused activation");
  }
}

void execute_kernel_CircleDepthwiseConv2D(const circle::Operator *cur_op,
                                          BaseRuntimeGraph *runtime_graph)
{
  const auto input_index = cur_op->inputs()->operator[](0);
  const auto weight_index = cur_op->inputs()->operator[](1);
  const auto bias_index = cur_op->inputs()->operator[](2);
  const auto output_index = cur_op->outputs()->operator[](0);

  assert(input_index != -1);
  assert(weight_index != -1);
  assert(output_index != -1);

  const auto input = runtime_graph->getCircleTensorByIndex(input_index);
  const auto weights = runtime_graph->getCircleTensorByIndex(weight_index);
  const auto bias = runtime_graph->getCircleTensorByIndex(bias_index);
  const auto output = runtime_graph->getCircleTensorByIndex(output_index);

  assert(input != nullptr);
  assert(weights != nullptr);
  assert(output != nullptr);

  const auto *options = cur_op->builtin_options_as_DepthwiseConv2DOptions();

  switch (Tensor::element_type(input))
  {
#ifndef DIS_FLOAT
    case DataType::FLOAT32:
      if (Tensor::element_type(weights) == DataType::FLOAT32)
      {
        evalFloat(input, weights, bias, output, options, runtime_graph);
        break;
      }
#endif // DIS_FLOAT
    default:
      assert(false && "Unsupported type.");
  }
}

} // namespace luci_interpreter
