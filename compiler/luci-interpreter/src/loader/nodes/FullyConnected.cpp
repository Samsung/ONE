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

#include "kernels/FullyConnected.h"

namespace luci_interpreter
{

std::unique_ptr<Kernel> build_kernel_CircleFullyConnected(const luci::CircleNode *circle_node,
                                                          KernelBuilderHelper &helper)
{
  const auto *node = loco::must_cast<const luci::CircleFullyConnected *>(circle_node);
  assert(node->arity() == 3);

  const Tensor *input = helper.getInputTensor(node->input());
  const Tensor *weights = helper.getInputTensor(node->weights());
  const Tensor *bias = helper.getOptionalInputTensor(node->bias());
  Tensor *output = helper.getOutputTensor(node);

  FullyConnectedParams params{};
  params.activation = node->fusedActivationFunction();
  params.keep_num_dims = node->keep_num_dims();
  if (weights->element_type() == loco::DataType::S4 ||
      weights->element_type() == loco::DataType::U4)
  {
    auto scratchpad =
      std::make_unique<Tensor>(input->element_type(), weights->shape(), AffineQuantization{}, "");
    scratchpad->set_observable(false);
    scratchpad->set_data_buffer(nullptr);
    Tensor *scratchpad_tmp =
      helper.getRuntimeGraph(node->graph())->addTensor(std::move(scratchpad));
    helper.getRuntimeGraph(node->graph())->configureAllocations(scratchpad_tmp);
    return std::make_unique<kernels::FullyConnected>(input, weights, bias, output, scratchpad_tmp,
                                                     params);
  }
  return std::make_unique<kernels::FullyConnected>(input, weights, bias, output, params);
}

} // namespace luci_interpreter
