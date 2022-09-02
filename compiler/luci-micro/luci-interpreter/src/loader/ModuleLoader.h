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
#include "luci_interpreter/MemoryManager.h"
#include "luci_interpreter/core/CircleMicroReader.h"

#include <luci/IR/Module.h>

#include <unordered_map>

namespace luci_interpreter
{

class ModuleLoader
{
public:
  ModuleLoader(const char *model_data_raw, RuntimeModule *runtime_module,
               IMemoryManager *memory_manager);

  void load();

private:
  IMemoryManager *_memory_manager;
  const char *_model_data_raw;
  RuntimeModule *_runtime_module;
  std::vector<RuntimeGraph *> _runtime_graphs;
  std::unique_ptr<std::unordered_map<int32_t, Tensor *>> _index_to_tensor;
};

} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_LOADER_MODULELOADER_H
