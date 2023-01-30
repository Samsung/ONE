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
    _memory_manager(memory_manager), _index_to_tensor(std::unordered_map<int32_t, Tensor *>{})
{
}

void ModuleLoader::load(bool use_static_memory_manager)
{
  const circle::Model *model = circle::GetModel(_model_data_raw);

  CircleReader reader;
  if (!reader.parse(model))
    assert(false && "Error during parse");

  for (size_t i = 0; i < reader.num_subgraph(); ++i)
  {
    _runtime_graphs.emplace_back(
      _runtime_module->addGraph(_memory_manager, use_static_memory_manager));
  }

  for (size_t i = 0; i < reader.num_subgraph(); ++i)
  {
    if (!reader.select_subgraph(i))
      assert(false && "Error during select subgraph");
    IBaseRuntimeGraph *runtime_graph = _runtime_graphs.at(i);
    GraphLoader loader(&reader, runtime_graph, _memory_manager, &_index_to_tensor);

    loader.initInputTensors(use_static_memory_manager);
    loader.loadTensors(use_static_memory_manager);
    loader.loadOperators(use_static_memory_manager);
  }

  // For Dynamic Memory manager we build memory allocate/deallocate plan and then configure kernels.
  // For Static Memory manager we only configure kernels.
  if (not use_static_memory_manager)
  {
    // Dynamic memory manager case
    for (size_t i = 0; i < reader.num_subgraph(); ++i)
    {
      IBaseRuntimeGraph *runtime_graph = _runtime_graphs.at(i);
      runtime_graph->configure();
    }
  }
  else
  {
    // Static memory manager case
    for (size_t i = 0; i < reader.num_subgraph(); ++i)
    {
      _runtime_graphs.at(i)->configure_kernels();
    }
  }
}

} // namespace luci_interpreter
