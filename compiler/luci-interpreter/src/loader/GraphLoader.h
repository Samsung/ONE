/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef LUCI_INTERPRETER_LOADER_GRAPHLOADER_H
#define LUCI_INTERPRETER_LOADER_GRAPHLOADER_H

#include "core/RuntimeGraph.h"
#include "loader/RuntimeToIR.h"

#include <loco/IR/Graph.h>

#include <unordered_map>

namespace luci_interpreter
{

class ModuleLoader;

class GraphLoader
{
public:
  GraphLoader(const ModuleLoader &module_loader, const loco::Graph *graph,
              RuntimeGraph *runtime_graph, RuntimeToIR &runtime_to_ir);

  void load();

  Tensor *getTensorForNode(const loco::Node *node) const { return _node_to_tensor.at(node); }

private:
  void loadOperators();
  void initInputOutputTensors() const;
  void loadTensors();

  const ModuleLoader &_module_loader;
  const loco::Graph *_graph;
  RuntimeGraph *_runtime_graph;
  RuntimeToIR &_runtime_to_ir;

  std::unordered_map<const loco::Node *, Tensor *> _node_to_tensor;
};

} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_LOADER_GRAPHLOADER_H
