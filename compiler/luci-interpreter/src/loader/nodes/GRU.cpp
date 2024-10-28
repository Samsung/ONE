/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/GRU.h"

namespace luci_interpreter
{

std::unique_ptr<Kernel> build_kernel_CircleGRU(const luci::CircleNode *circle_node,
                                               KernelBuilderHelper &helper)
{
  const auto *node = loco::must_cast<const luci::CircleGRU *>(circle_node);
  assert(node->arity() == 6);

  const Tensor *input = helper.getInputTensor(node->input());
  const Tensor *hidden_hidden = helper.getInputTensor(node->hidden_hidden());
  const Tensor *hidden_hidden_bias = helper.getInputTensor(node->hidden_hidden_bias());
  const Tensor *hidden_input = helper.getInputTensor(node->hidden_input());
  const Tensor *hidden_input_bias = helper.getInputTensor(node->hidden_input_bias());
  const Tensor *state = helper.getInputTensor(node->state());

  Tensor *output = helper.getOutputTensor(node);

  GRUParams params{};
  params.fused_act_function = node->fusedActivationFunction();
  params.return_sequences = node->returnSequences();
  params.time_major = node->timeMajor();

  return std::make_unique<kernels::GRU>(input, hidden_hidden, hidden_hidden_bias, hidden_input,
                                        hidden_input_bias, state, output, params);
}

} // namespace luci_interpreter
