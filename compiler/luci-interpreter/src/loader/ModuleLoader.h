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

#ifndef LUCI_INTERPRETER_LOADER_MODULELOADER_H
#define LUCI_INTERPRETER_LOADER_MODULELOADER_H

#include "core/RuntimeModule.h"
#include "loader/RuntimeToIR.h"
#include "luci_interpreter/MemoryManager.h"

#include <luci/IR/Module.h>

#include <unordered_map>

namespace luci_interpreter
{

class ModuleLoader
{
public:
  ModuleLoader(const luci::Module *module, RuntimeModule *runtime_module,
               RuntimeToIR &runtime_to_ir,
               std::unordered_map<const loco::Node *, Tensor *> &node_to_tensor,
               IMemoryManager *memory_manager);

  void load();

private:
  IMemoryManager *_memory_manager;
  const luci::Module *_module;
  RuntimeModule *_runtime_module;
  RuntimeToIR &_runtime_to_ir;
  std::unordered_map<const loco::Node *, Tensor *> &_node_to_tensor;
  std::unordered_map<const loco::Graph *, RuntimeGraph *> _graph_to_runtime_graph;
};

} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_LOADER_MODULELOADER_H
