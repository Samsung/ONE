/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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
#include "TISOKernel.h"
#include "PALGRUCell.h"

#include "PALBroadcastTo.h"

namespace luci_interpreter
{
namespace
{
constexpr int kMaxDims = 5;
} // namespace

void configure_kernel_CircleCustom(const circle::Operator *cur_op,
                                        BaseRuntimeGraph *runtime_graph)
{
  const auto input_index = cur_op->inputs()->operator[](0);
  const auto weight_input_index = cur_op->inputs()->operator[](1);
  const auto weight_hidden_index = cur_op->inputs()->operator[](2);
  const auto bias_input_index = cur_op->inputs()->operator[](3);
  const auto bias_hidden_index = cur_op->inputs()->operator[](4);
  const auto hidden_state_index = cur_op->inputs()->operator[](5);
  const auto output_index = cur_op->outputs()->operator[](0);

  assert(input_index != -1);
  assert(weight_input_index != -1);
  assert(weight_hidden_index != -1);
  assert(bias_input_index != -1);
  assert(bias_hidden_index != -1);
  assert(hidden_state_index != -1);
  assert(output_index != -1);

  const auto input_tensor = runtime_graph->getCircleTensorByIndex(input_index);
  const auto weight_input_tensor = runtime_graph->getCircleTensorByIndex(weight_input_index);
  const auto weight_hidden_tensor = runtime_graph->getCircleTensorByIndex(weight_hidden_index);
  const auto bias_input_tensor = runtime_graph->getCircleTensorByIndex(bias_input_index);
  const auto bias_hidden_tensor = runtime_graph->getCircleTensorByIndex(bias_hidden_index);
  const auto hidden_state_tensor = runtime_graph->getCircleTensorByIndex(hidden_state_index);
  const auto output = runtime_graph->getCircleTensorByIndex(output_index);

  assert(input_tensor != nullptr);
  assert(weight_input_tensor != nullptr);
  assert(weight_hidden_tensor != nullptr);
  assert(bias_input_tensor != nullptr);
  assert(bias_hidden_tensor != nullptr);
  assert(hidden_state_tensor != nullptr);
  assert(output != nullptr);

 // TODO: add checks

}

void execute_kernel_CircleCustom(const circle::Operator *cur_op,
                                      BaseRuntimeGraph *runtime_graph)
{
  const auto input_index = cur_op->inputs()->operator[](0);
  const auto weight_input_index = cur_op->inputs()->operator[](1);
  const auto weight_hidden_index = cur_op->inputs()->operator[](2);
  const auto bias_input_index = cur_op->inputs()->operator[](3);
  const auto bias_hidden_index = cur_op->inputs()->operator[](4);
  const auto hidden_state_index = cur_op->inputs()->operator[](5);
  const auto output_index = cur_op->outputs()->operator[](0);

  assert(input_index != -1);
  assert(weight_input_index != -1);
  assert(weight_hidden_index != -1);
  assert(bias_input_index != -1);
  assert(bias_hidden_index != -1);
  assert(hidden_state_index != -1);
  assert(output_index != -1);

  const auto input_tensor = runtime_graph->getCircleTensorByIndex(input_index);
  const auto weight_input_tensor = runtime_graph->getCircleTensorByIndex(weight_input_index);
  const auto weight_hidden_tensor = runtime_graph->getCircleTensorByIndex(weight_hidden_index);
  const auto bias_input_tensor = runtime_graph->getCircleTensorByIndex(bias_input_index);
  const auto bias_hidden_tensor = runtime_graph->getCircleTensorByIndex(bias_hidden_index);
  const auto hidden_state_tensor = runtime_graph->getCircleTensorByIndex(hidden_state_index);
  const auto output = runtime_graph->getCircleTensorByIndex(output_index);

  assert(input_tensor != nullptr);
  assert(weight_input_tensor != nullptr);
  assert(weight_hidden_tensor != nullptr);
  assert(bias_input_tensor != nullptr);
  assert(bias_hidden_tensor != nullptr);
  assert(hidden_state_tensor != nullptr);
  assert(output != nullptr);

  float *input_data = kernels::getTensorData<float>(runtime_graph->getDataByTensor(input_tensor));
  float *weight_input_data = kernels::getTensorData<float>(runtime_graph->getConstDataByTensor(weight_input_tensor));
  float *weight_hidden_data = kernels::getTensorData<float>(runtime_graph->getConstDataByTensor(weight_hidden_tensor));
  float *bias_hidden_data = kernels::getTensorData<float>(runtime_graph->getConstDataByTensor(bias_hidden_tensor));
  float *bias_input_data = kernels::getTensorData<float>(runtime_graph->getConstDataByTensor(bias_input_tensor));
  float *hidden_state_data = kernels::getTensorData<float>(runtime_graph->getConstDataByTensor(hidden_state_tensor));
  float *output_data = kernels::getTensorData<float>(runtime_graph->getDataByTensor(output));

  auto input_shape = Tensor::tensor_shape(input_tensor);
  auto weight_input_shape = Tensor::tensor_shape(weight_input_tensor);
  auto weight_hidden_shape = Tensor::tensor_shape(weight_hidden_tensor);
  auto hidden_state_shape = Tensor::tensor_shape(hidden_state_tensor);
  auto output_shape = Tensor::tensor_shape(output);

  luci_interpreter_pal::GRU(input_data, weight_input_data, weight_hidden_data,
                            bias_input_data, bias_hidden_data, hidden_state_data,
                            output_data, input_shape.data(), output_shape.data(),
                            weight_input_shape.data(), weight_hidden_shape.data());

}
} // namespace luci_interpreter
