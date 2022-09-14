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
ModuleLoader::ModuleLoader(const char *model_data_raw, RuntimeModule *runtime_module,
                           IMemoryManager *memory_manager)
  : _model_data_raw(model_data_raw), _runtime_module(runtime_module),
    _memory_manager(memory_manager),
    _index_to_tensor(std::make_unique<std::map<int32_t, Tensor *>>())
{
}

void ModuleLoader::load()
{
  const circle::Model *model = circle::GetModel(_model_data_raw);

  CircleReader reader;
  if (!reader.parse(model))
    throw std::runtime_error("Error during parse");

  for (size_t i = 0; i < reader.num_subgraph(); ++i)
  {
    _runtime_graphs.emplace_back(_runtime_module->addGraph(_memory_manager));
  }

  for (size_t i = 0; i < reader.num_subgraph(); ++i)
  {
    if (!reader.select_subgraph(i))
      throw std::runtime_error("Error during select subgraph");
    RuntimeGraph *runtime_graph = _runtime_graphs.at(i);
    GraphLoader loader(&reader, runtime_graph, _memory_manager, _index_to_tensor.get());

    loader.initInputTensors();
    loader.loadTensors();
    loader.loadOperators();
  }

  for (size_t i = 0; i < reader.num_subgraph(); ++i)
  {
    RuntimeGraph *runtime_graph = _runtime_graphs.at(i);
    runtime_graph->configure();
  }
}

} // namespace luci_interpreter
