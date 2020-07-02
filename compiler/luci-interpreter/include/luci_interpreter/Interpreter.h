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

#ifndef LUCI_INTERPRETER_INTERPRETER_H
#define LUCI_INTERPRETER_INTERPRETER_H

#include "luci_interpreter/core/Tensor.h"

#include <luci/IR/Nodes/CircleInput.h>
#include <luci/IR/Nodes/CircleOutput.h>

#include <luci/IR/Module.h>

#include <memory>
#include <vector>

namespace luci_interpreter
{

class ExecutionObserver
{
public:
  virtual ~ExecutionObserver();

  // Called when the value of a tensor has been updated during execution.
  virtual void postTensorWrite(const luci::CircleNode *node, const Tensor *tensor);
};

class Interpreter
{
public:
  explicit Interpreter(const luci::Module *module);

  ~Interpreter();

  void writeInputTensor(const luci::CircleInput *input_node, const void *data, size_t data_size);

  void readOutputTensor(const luci::CircleOutput *output_node, void *data, size_t data_size);

  void interpret();

  void attachObserver(ExecutionObserver *observer);

private:
  std::unique_ptr<class RuntimeModule> _runtime_module;

  // Observer functionality support.
  std::unique_ptr<struct RuntimeToIR> _runtime_to_ir;
  std::unique_ptr<class EventNotifier> _event_notifier;
  std::vector<ExecutionObserver *> _observers;
};

} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_INTERPRETER_H
