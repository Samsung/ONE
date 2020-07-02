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

#include "ModuleLoader.h"

#include "GraphLoader.h"

namespace luci_interpreter
{

ModuleLoader::ModuleLoader(const luci::Module *module, RuntimeModule *runtime_module,
                           RuntimeToIR &runtime_to_ir)
    : _module(module), _runtime_module(runtime_module), _runtime_to_ir(runtime_to_ir)
{
}

void ModuleLoader::load()
{
  // Runtime graphs have to be created in advance, because they will be needed during the loading
  // process for control flow nodes.
  for (size_t i = 0; i < _module->size(); ++i)
  {
    _graph_to_runtime_graph.emplace(_module->graph(i), _runtime_module->addGraph());
  }
  for (size_t i = 0; i < _module->size(); ++i)
  {
    const loco::Graph *graph = _module->graph(i);
    RuntimeGraph *runtime_graph = _graph_to_runtime_graph.at(graph);
    GraphLoader loader(*this, graph, runtime_graph, _runtime_to_ir);
    loader.load();
  }
}

} // namespace luci_interpreter
