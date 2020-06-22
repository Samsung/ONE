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
#include "core/EventNotifier.h"

#include <memory>
#include <vector>

namespace luci_interpreter
{

class RuntimeModule
{
public:
  explicit RuntimeModule(EventNotifier *event_notifier) : _event_notifier(event_notifier) {}

  EventNotifier *getEventNotifier() const { return _event_notifier; }

  RuntimeGraph *addGraph()
  {
    _graphs.push_back(std::make_unique<RuntimeGraph>(this));
    return _graphs.back().get();
  }

  const std::vector<Tensor *> &getInputTensors() const { return getMainGraph()->getInputTensors(); }
  const std::vector<Tensor *> &getOutputTensors() const
  {
    return getMainGraph()->getOutputTensors();
  }

  void configure() { getMainGraph()->configure(); }

  void execute() const { getMainGraph()->execute(); }

private:
  RuntimeGraph *getMainGraph() const { return _graphs[0].get(); }

  EventNotifier *const _event_notifier;
  std::vector<std::unique_ptr<RuntimeGraph>> _graphs;
};

} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_CORE_RUNTIMEMODULE_H
