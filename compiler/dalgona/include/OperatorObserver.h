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

#ifndef __DALGONA_OPERATOR_OBSERVER_H__
#define __DALGONA_OPERATOR_OBSERVER_H__

#include <loco/IR/Graph.h>
#include <luci_interpreter/Interpreter.h>
#include <luci_interpreter/core/Tensor.h>

#include <pybind11/embed.h>

#include <vector>
#include <unordered_map>

namespace py = pybind11;

namespace dalgona
{

class OperatorObserver : public luci_interpreter::ExecutionObserver
{
public:
  OperatorObserver(luci_interpreter::Interpreter *interpreter) : _interpreter(interpreter)
  {
    // Do nothing
  }

  // Called when the analysis starts
  void importAnalysis(const std::string &analysis_path, py::object &globals,
                      const std::string &analysis_args);

  // Called after the analysis is done
  void endAnalysis();

  // Called before a network is started to be executed
  void startNetworkExecution(loco::Graph *graph);

  // Called after a network is executed
  void endNetworkExecution(loco::Graph *graph);

  // Called before an operator is executed
  void preOperatorExecute(const luci::CircleNode *node) override {}

  // Called after an operator is executed
  void postOperatorExecute(const luci::CircleNode *node) override {}

private:
  luci_interpreter::Interpreter *_interpreter{nullptr};
  py::object _analysis;
};

} // namespace dalgona

#endif // __DALGONA_OPERATOR_OBSERVER_H__
