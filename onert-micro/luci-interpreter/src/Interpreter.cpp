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
#include "luci_interpreter/memory_managers/SimpleMemoryManager.h"

#include "loader/ModuleLoader.h"

namespace luci_interpreter
{

Interpreter::Interpreter(const char *model_data_raw, bool allocate_input)
{
  _runtime_module = std::make_unique<RuntimeModule>();

  _default_memory_manager = std::make_unique<SimpleMemoryManager>();
  _default_memory_manager->is_allocate_input(allocate_input);

  ModuleLoader loader(model_data_raw, _runtime_module.get(), _default_memory_manager.get());
  loader.load();
}

Interpreter::Interpreter(const char *model_data_raw, IMemoryManager *memory_manager,
                         bool allocate_input)
{
  assert(memory_manager && "Use Interpreter::Interpreter(module) constructor instead");
  _runtime_module = std::make_unique<RuntimeModule>();

  memory_manager->is_allocate_input(allocate_input);

  ModuleLoader loader(model_data_raw, _runtime_module.get(), memory_manager);
  loader.load();
}

Interpreter::~Interpreter() = default;

void Interpreter::interpret() { _runtime_module->execute(); }

std::vector<Tensor *> Interpreter::getInputTensors() { return _runtime_module->getInputTensors(); }

std::vector<Tensor *> Interpreter::getOutputTensors()
{
  return _runtime_module->getOutputTensors();
}

void Interpreter::writeInputTensor(Tensor *input_tensor, const void *data, size_t data_size)
{
  if (data != nullptr)
    input_tensor->writeData(data, data_size);
}

void Interpreter::writeInputTensorWithoutCopy(Tensor *input_tensor, const void *data)
{
  if (data != nullptr)
    input_tensor->writeDataWithoutCopy(const_cast<void *>(data));
}

void Interpreter::readOutputTensor(const Tensor *output_tensor, void *data, size_t data_size)
{
  if (data != nullptr)
    output_tensor->readData(data, data_size);
}

} // namespace luci_interpreter
