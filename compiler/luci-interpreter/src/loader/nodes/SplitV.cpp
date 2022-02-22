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

#include "kernels/SplitV.h"

namespace luci_interpreter
{

std::unique_ptr<Kernel> build_kernel_CircleSplitV(const luci::CircleNode *circle_node,
                                                  KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleSplitV *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  auto output_nodes = collectOutputNodes<luci::CircleSplitVOut>(node);
  assert(node->arity() == 3);
  assert(output_nodes.size() == static_cast<size_t>(node->num_split()));

  const Tensor *input = helper.getInputTensor(node->input());
  const Tensor *sizes_data = helper.getInputTensor(node->size_splits());
  const Tensor *axis = helper.getInputTensor(node->split_dim());
  std::vector<Tensor *> outputs = helper.getOutputTensors(output_nodes);

  // NOTE 'num_splits' attribute is ignored.
  return std::make_unique<kernels::SplitV>(input, sizes_data, axis, std::move(outputs));
}

} // namespace luci_interpreter
