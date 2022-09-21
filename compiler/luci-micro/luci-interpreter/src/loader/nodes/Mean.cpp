/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/Mean.h"

namespace luci_interpreter
{

std::unique_ptr<Kernel> build_kernel_CircleMean(std::vector<const Tensor *> &&inputs,
                                                std::vector<Tensor *> &&outputs,
                                                const uint32_t op_index, KernelBuilder &builder)
{
  assert(inputs.size() == 2);

  const Tensor *input = inputs.at(0);
  const Tensor *axis = inputs.at(1);
  Tensor *output = outputs.at(0);

  auto temp_index_unique = std::make_unique<Tensor>(DataType::S32, Shape({}), nullptr);
  temp_index_unique->set_data_buffer(nullptr);
  Tensor *temp_index = builder.get_runtime_graph()->addTensor(std::move(temp_index_unique));

  auto resolved_axes_unique = std::make_unique<Tensor>(DataType::S32, Shape({}), nullptr);
  resolved_axes_unique->set_data_buffer(nullptr);
  Tensor *resolved_axes = builder.get_runtime_graph()->addTensor(std::move(resolved_axes_unique));

  auto temp_sum_unique = std::make_unique<Tensor>(input->element_type(), Shape({}), nullptr);
  temp_sum_unique->set_data_buffer(nullptr);
  Tensor *temp_sum = builder.get_runtime_graph()->addTensor(std::move(temp_sum_unique));

  circle::OperatorT oper_t;
  builder.get_circle_reader()->operators()[op_index]->UnPackTo(&oper_t);
  const auto *options = oper_t.builtin_options.AsReducerOptions();

  ReducerParams params{};
  params.keep_dims = options->keep_dims;

  return std::make_unique<kernels::Mean>(input, axis, output, temp_index, resolved_axes, temp_sum,
                                         params);
}

} // namespace luci_interpreter
