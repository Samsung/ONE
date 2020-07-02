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

#include "luci_interpreter/Interpreter.h"

#include "loader/ModuleLoader.h"

#include <stdexcept>

namespace luci_interpreter
{

namespace
{

class EventNotifierImpl final : public EventNotifier
{
public:
  EventNotifierImpl(RuntimeToIR &runtime_to_ir, const std::vector<ExecutionObserver *> &observers)
      : _runtime_to_ir(runtime_to_ir), _observers(observers)
  {
  }

  void postTensorWrite(Tensor *tensor) const override
  {
    assert(tensor != nullptr);
    for (const auto &observer : _observers)
    {
      observer->postTensorWrite(_runtime_to_ir.tensor_to_node.at(tensor), tensor);
    }
  }

private:
  RuntimeToIR &_runtime_to_ir;
  const std::vector<ExecutionObserver *> &_observers;
};

} // namespace

Interpreter::Interpreter(const luci::Module *module)
{
  _runtime_to_ir = std::make_unique<RuntimeToIR>();
  _event_notifier = std::make_unique<EventNotifierImpl>(*_runtime_to_ir, _observers);
  _runtime_module = std::make_unique<RuntimeModule>(_event_notifier.get());
  ModuleLoader loader(module, _runtime_module.get(), *_runtime_to_ir);
  loader.load();

  _runtime_module->configure();
}

Interpreter::~Interpreter() = default;

void Interpreter::writeInputTensor(const luci::CircleInput *input_node, const void *data,
                                   size_t data_size)
{
  Tensor *tensor = _runtime_module->getInputTensors()[input_node->index()];
  if (tensor == nullptr)
  {
    const std::string &name = input_node->name();
    throw std::runtime_error("Cannot find tensor for input node named \"" + name + "\".");
  }
  tensor->writeData(data, data_size);
}

void Interpreter::readOutputTensor(const luci::CircleOutput *output_node, void *data,
                                   size_t data_size)
{
  Tensor *tensor = _runtime_module->getOutputTensors()[output_node->index()];
  if (tensor == nullptr)
  {
    const std::string &name = output_node->name();
    throw std::runtime_error("Cannot find tensor for output node named \"" + name + "\".");
  }
  tensor->readData(data, data_size);
}

void Interpreter::interpret() { _runtime_module->execute(); }

void Interpreter::attachObserver(ExecutionObserver *observer)
{
  if (std::find(_observers.cbegin(), _observers.cend(), observer) != _observers.cend())
    throw std::runtime_error("Observer is already attached.");
  _observers.push_back(observer);
}

ExecutionObserver::~ExecutionObserver() = default;

void ExecutionObserver::postTensorWrite(const luci::CircleNode *, const Tensor *) {}

} // namespace luci_interpreter
