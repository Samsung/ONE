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

#ifndef LUCI_INTERPRETER_CORE_RUNTIMEMODULE_H
#define LUCI_INTERPRETER_CORE_RUNTIMEMODULE_H

#include "core/RuntimeGraph.h"
#include "luci_interpreter/MemoryManager.h"
#include "luci_interpreter/core/CircleMicroReader.h"

#include <memory>
#include <vector>

namespace luci_interpreter
{

class RuntimeModule
{
public:
  RuntimeModule() : _circle_micro_reader(std::make_unique<CircleReader>()) {}

  RuntimeGraph *addGraph(IMemoryManager *memory_manager)
  {
    _graphs.push_back(std::make_unique<RuntimeGraph>(this, memory_manager));
    return _graphs.back().get();
  }

  const std::vector<Tensor *> &getInputTensors() const { return getMainGraph()->getInputTensors(); }
  const std::vector<Tensor *> &getOutputTensors() const
  {
    return getMainGraph()->getOutputTensors();
  }

  void execute() const { getMainGraph()->execute(); }

  CircleReader *getCircleMicroReader() { return _circle_micro_reader.get(); }

private:
  RuntimeGraph *getMainGraph() const { return _graphs[0].get(); }

  std::vector<std::unique_ptr<RuntimeGraph>> _graphs;
  std::unique_ptr<CircleReader> _circle_micro_reader;
};

} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_CORE_RUNTIMEMODULE_H
