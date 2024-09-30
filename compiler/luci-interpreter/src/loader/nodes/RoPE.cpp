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

#include "kernels/RoPE.h"

namespace luci_interpreter
{

std::unique_ptr<Kernel> build_kernel_CircleRoPE(const luci::CircleNode *circle_node,
                                                KernelBuilderHelper &helper)
{
  const auto *node = loco::must_cast<const luci::CircleRoPE *>(circle_node);
  assert(node->arity() == 3);

  const Tensor *input = helper.getInputTensor(node->input());
  const Tensor *sin_table = helper.getInputTensor(node->sin_table());
  const Tensor *cos_table = helper.getInputTensor(node->cos_table());

  Tensor *output = helper.getOutputTensor(node);

  RoPEParams params{};
  params.mode = node->mode();

  return std::make_unique<kernels::RoPE>(input, sin_table, cos_table, output, params);
}

} // namespace luci_interpreter
