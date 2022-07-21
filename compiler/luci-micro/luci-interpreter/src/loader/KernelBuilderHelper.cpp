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

#include "loader/KernelBuilderHelper.h"

#include <luci/IR/Nodes/CircleOutput.h>

namespace luci_interpreter
{

const Tensor *KernelBuilderHelper::getInputTensor(const loco::Node *node) const
{
  const Tensor *tensor = _node_to_tensor.at(node);
  assert(tensor != nullptr);
  return tensor;
}

const Tensor *KernelBuilderHelper::getOptionalInputTensor(const loco::Node *node) const
{
  if (dynamic_cast<const luci::CircleOutputExclude *>(node))
  {
    return nullptr;
  }
  return getInputTensor(node);
}

Tensor *KernelBuilderHelper::getOutputTensor(const loco::Node *node) const
{
  Tensor *tensor = _node_to_tensor.at(node);
  assert(tensor != nullptr);
  return tensor;
}

std::vector<Tensor *>
KernelBuilderHelper::getOutputTensors(const std::vector<const loco::Node *> &nodes) const
{
  std::vector<Tensor *> tensors;
  tensors.reserve(nodes.size());
  for (const loco::Node *node : nodes)
    tensors.push_back(getOutputTensor(node));
  return tensors;
}

RuntimeGraph *KernelBuilderHelper::getRuntimeGraph(const loco::Graph *graph) const
{
  RuntimeGraph *runtime_graph = _graph_to_runtime_graph.at(graph);
  assert(runtime_graph != nullptr);
  return runtime_graph;
}

} // namespace luci_interpreter
