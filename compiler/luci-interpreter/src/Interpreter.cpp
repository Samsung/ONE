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
#include "luci_interpreter/SimpleMemoryManager.h"

#include "loader/ModuleLoader.h"

#include <stdexcept>

namespace luci_interpreter
{

namespace
{

class EventNotifierImpl final : public EventNotifier
{
public:
  EventNotifierImpl(const RuntimeToIR &runtime_to_ir,
                    const std::vector<ExecutionObserver *> &observers)
    : _runtime_to_ir(runtime_to_ir), _observers(observers)
  {
  }

  void postTensorWrite(const Tensor *tensor) override
  {
    assert(tensor != nullptr);
    for (const auto &observer : _observers)
    {
      observer->postTensorWrite(_runtime_to_ir.tensor_to_node.at(tensor), tensor);
    }
  }

  void preOperatorExecute(const Kernel *kernel) override
  {
    assert(kernel != nullptr);
    for (const auto &observer : _observers)
    {
      observer->preOperatorExecute(_runtime_to_ir.kernel_to_node.at(kernel));
    }
  }

  void postOperatorExecute(const Kernel *kernel) override
  {
    assert(kernel != nullptr);
    for (const auto &observer : _observers)
    {
      observer->postOperatorExecute(_runtime_to_ir.kernel_to_node.at(kernel));
    }
  }

private:
  const RuntimeToIR &_runtime_to_ir;
  const std::vector<ExecutionObserver *> &_observers;
};

} // namespace

Interpreter::Interpreter(const luci::Module *module)
{
  _runtime_to_ir = std::make_unique<RuntimeToIR>();
  _event_notifier = std::make_unique<EventNotifierImpl>(*_runtime_to_ir, _observers);
  _runtime_module = std::make_unique<RuntimeModule>(_event_notifier.get());

  _default_memory_manager = std::make_unique<SimpleMemoryManager>();

  ModuleLoader loader(module, _runtime_module.get(), *_runtime_to_ir, _node_to_tensor,
                      _default_memory_manager.get());
  loader.load();
}

Interpreter::Interpreter(const luci::Module *module,
                         luci_interpreter::IMemoryManager *memory_manager)
{
  assert(memory_manager && "Use Interpreter::Interpreter(module) constructor instead");

  _runtime_to_ir = std::make_unique<RuntimeToIR>();
  _event_notifier = std::make_unique<EventNotifierImpl>(*_runtime_to_ir, _observers);
  _runtime_module = std::make_unique<RuntimeModule>(_event_notifier.get());

  ModuleLoader loader(module, _runtime_module.get(), *_runtime_to_ir, _node_to_tensor,
                      memory_manager);
  loader.load();
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
  if (data != nullptr)
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
  if (data != nullptr)
    tensor->readData(data, data_size);
}

size_t Interpreter::getOutputTensorSize(const luci::CircleOutput *output_node)
{
  Tensor *tensor = _runtime_module->getOutputTensors()[output_node->index()];
  if (tensor == nullptr)
  {
    const std::string &name = output_node->name();
    throw std::runtime_error("Cannot find tensor size for output node named \"" + name + "\".");
  }

  size_t tensor_size = luci::size(output_node->dtype());
  for (int i = 0; i < tensor->shape().num_dims(); i++)
    tensor_size *= tensor->shape().dim(i);
  return tensor_size;
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

void ExecutionObserver::preOperatorExecute(const luci::CircleNode *) {}

void ExecutionObserver::postOperatorExecute(const luci::CircleNode *) {}

} // namespace luci_interpreter
