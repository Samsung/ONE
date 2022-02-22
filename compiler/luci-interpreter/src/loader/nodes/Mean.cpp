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

std::unique_ptr<Kernel> build_kernel_CircleMean(const luci::CircleNode *circle_node,
                                                KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleMean *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 2);

  const Tensor *input = helper.getInputTensor(node->input());
  const Tensor *axes = helper.getInputTensor(node->reduction_indices());
  Tensor *output = helper.getOutputTensor(node);

  auto temp_index_unique =
    std::make_unique<Tensor>(DataType::S32, Shape({}), AffineQuantization{}, "");
  temp_index_unique->set_observable(false);
  temp_index_unique->set_data_buffer(nullptr);
  Tensor *temp_index =
    helper.getRuntimeGraph(node->graph())->addTensor(std::move(temp_index_unique));

  auto resolved_axes_unique =
    std::make_unique<Tensor>(DataType::S32, Shape({}), AffineQuantization{}, "");
  resolved_axes_unique->set_observable(false);
  resolved_axes_unique->set_data_buffer(nullptr);
  Tensor *resolved_axes =
    helper.getRuntimeGraph(node->graph())->addTensor(std::move(resolved_axes_unique));

  auto temp_sum_unique =
    std::make_unique<Tensor>(input->element_type(), Shape({}), AffineQuantization{}, "");
  temp_sum_unique->set_observable(false);
  temp_sum_unique->set_data_buffer(nullptr);
  Tensor *temp_sum = helper.getRuntimeGraph(node->graph())->addTensor(std::move(temp_sum_unique));

  ReducerParams params{};
  params.keep_dims = node->keep_dims();

  return std::make_unique<kernels::Mean>(input, axes, output, temp_index, resolved_axes, temp_sum,
                                         params);
}

} // namespace luci_interpreter
