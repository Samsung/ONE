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

#ifndef LUCI_INTERPRETER_LOADER_KERNELBUILDER_HELPER_H
#define LUCI_INTERPRETER_LOADER_KERNELBUILDER_HELPER_H

#include "core/Kernel.h"
#include "core/RuntimeGraph.h"

#include <loco/IR/Graph.h>
#include <loco/IR/Node.h>

#include <vector>
#include <unordered_map>

namespace luci_interpreter
{

class KernelBuilderHelper
{
public:
  KernelBuilderHelper(
    const std::unordered_map<const loco::Graph *, RuntimeGraph *> &graph_to_runtime_graph,
    const std::unordered_map<const loco::Node *, Tensor *> &node_to_tensor)
    : _graph_to_runtime_graph(graph_to_runtime_graph), _node_to_tensor(node_to_tensor)
  {
  }

protected:
  const Tensor *getInputTensor(const loco::Node *node) const;
  const Tensor *getOptionalInputTensor(const loco::Node *node) const;

  Tensor *getOutputTensor(const loco::Node *node) const;
  std::vector<Tensor *> getOutputTensors(const std::vector<const loco::Node *> &nodes) const;

  RuntimeGraph *getRuntimeGraph(const loco::Graph *graph) const;

private:
  const std::unordered_map<const loco::Graph *, RuntimeGraph *> &_graph_to_runtime_graph;
  const std::unordered_map<const loco::Node *, Tensor *> &_node_to_tensor;
};

} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_LOADER_KERNELBUILDER_HELPER_H
