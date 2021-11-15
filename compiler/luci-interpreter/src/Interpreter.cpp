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
#include "loader/GraphLoader.h"
#include "loader/ModuleLoader.h"
#include "loader/KernelBuilder.h"
#include <loco/IR/Algorithm.h>
#include <iostream>
#include <iostream>
#include <stdexcept>

namespace luci_interpreter
{

namespace
{
bool isExecutableNode(const luci::CircleNode *node)
{
  switch (node->opcode())
  {
    // These nodes denote inputs / outputs of a graph.
    case luci::CircleOpcode::CIRCLECONST:
    case luci::CircleOpcode::CIRCLEINPUT:
    case luci::CircleOpcode::CIRCLEOUTPUT:
    case luci::CircleOpcode::CIRCLEOUTPUTEXCLUDE:
      // The following nodes denote outputs of multiple-output nodes.
    case luci::CircleOpcode::CIRCLEIFOUT:
    case luci::CircleOpcode::CIRCLESPLITOUT:
    case luci::CircleOpcode::CIRCLESPLITVOUT:
    case luci::CircleOpcode::CIRCLEUNPACKOUT:
    case luci::CircleOpcode::CIRCLEWHILEOUT:
      return false;
    default:
      return true;
  }
}
bool isTensorProducingNode(const luci::CircleNode *node)
{
  switch (node->opcode())
  {
    // Output nodes do not produce tensors.
    case luci::CircleOpcode::CIRCLEOUTPUT:
      // The following nodes are multiple-output nodes. They do not produce tensors, the tensors
      // are produced by the corresponding *Out nodes instead.
    case luci::CircleOpcode::IF:
    case luci::CircleOpcode::SPLIT:
    case luci::CircleOpcode::UNPACK:
      return false;
    default:
      return true;
  }
}
template <typename NodeT> Shape getNodeShape(const NodeT *node)
{
  Shape shape(node->rank());
  for (uint32_t i = 0; i < node->rank(); ++i)
  {
    shape.dim(i) = node->dim(i).value();
  }
  return shape;
}
template <DataType DT> const void *getNodeDataImpl(const luci::CircleConst *node, size_t *data_size)
{
  const size_t element_size = getDataTypeSize(DT);
  const int32_t num_elements = node->size<DT>();

  *data_size = num_elements * element_size;
  if (*data_size > 0)
  {
    // FIXME There is no good way to get the pointer to the data currently.
    return &node->at<DT>(0);
  }
  return nullptr;
}

const void *getNodeData(const luci::CircleConst *node, size_t *data_size)
{
  switch (node->dtype())
  {
    case DataType::U8:
      return getNodeDataImpl<DataType::U8>(node, data_size);
    case DataType::FLOAT32:
      return getNodeDataImpl<DataType::FLOAT32>(node, data_size);
    case DataType::S8:
      return getNodeDataImpl<DataType::S8>(node, data_size);
    case DataType::S16:
      return getNodeDataImpl<DataType::S16>(node, data_size);
    case DataType::S32:
      return getNodeDataImpl<DataType::S32>(node, data_size);
    case DataType::S64:
      return getNodeDataImpl<DataType::S64>(node, data_size);
    case DataType::BOOL:
      return getNodeDataImpl<DataType::BOOL>(node, data_size);
    default:
      throw std::runtime_error("Unsupported type.");
  }
}
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

Interpreter::Interpreter(const luci::Module *module,
                         luci_interpreter::IMemoryManager *memory_manager)
{
  _runtime_to_ir = std::make_unique<RuntimeToIR>();
  _event_notifier = std::make_unique<EventNotifierImpl>(*_runtime_to_ir, _observers);
  _runtime_module = std::make_unique<RuntimeModule>(_event_notifier.get());

  if (memory_manager == nullptr)
  {
    _default_memory_manager = std::make_unique<SimpleMemoryManager>();
    _memory_manager = _default_memory_manager.get();
  }
  else
  {
    _memory_manager = memory_manager;
  }
  ModuleLoader loader(module, _runtime_module.get(), *_runtime_to_ir, _node_to_tensor,
                      _memory_manager);
  loader.load();
}

Interpreter::~Interpreter() = default;

void Interpreter::writeInputTensor(const luci::CircleInput *input_node, const void *data,
                                   size_t data_size)
{
  auto input_tensors = _runtime_module->getInputTensors();
  Tensor *tensor = input_tensors[input_node->index()];

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
