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

#include "While.h"

namespace luci_interpreter
{

std::unique_ptr<Kernel> build_kernel_CircleWhile(const luci::CircleNode *circle_node,
                                                 KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleWhile *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");

  auto output_nodes = collectOutputNodes<luci::CircleWhileOut>(node);
  assert(node->arity() == node->input_count());
  assert(output_nodes.size() == static_cast<size_t>(node->output_count()));

  std::vector<const Tensor *> inputs(node->input_count());
  for (uint32_t i = 0; i < node->input_count(); ++i)
  {
    inputs[i] = helper.getInputTensor(node->input(i));
  }
  std::vector<Tensor *> outputs = helper.getOutputTensors(output_nodes);

  RuntimeGraph *cond_graph = helper.getRuntimeGraph(node->cond_graph());
  RuntimeGraph *body_graph = helper.getRuntimeGraph(node->body_graph());

  return std::make_unique<kernels::While>(std::move(inputs), std::move(outputs), cond_graph,
                                          body_graph);
}

} // namespace luci_interpreter
