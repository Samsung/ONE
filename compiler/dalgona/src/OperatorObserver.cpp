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

#include "OperatorObserver.h"
#include "Utils.h"

#include <loco/IR/Graph.h>

namespace dalgona
{

void OperatorObserver::importAnalysis(const std::string &analysis_path, py::object &globals,
                                      const std::string &analysis_args)
{
  auto base_filename = analysis_path.substr(analysis_path.find_last_of("/\\") + 1);
  // module name must be the same with the python code
  // ex: MyAnalysis.py -> module name = MyAnalysis
  auto module_name = base_filename.substr(0, base_filename.find_last_of('.'));

  py::dict locals;
  locals["path"] = py::cast(analysis_path);

  py::eval<py::eval_statements>("import sys\n"
                                "import os\n"
                                "sys.path.append(os.path.dirname(path))\n"
                                "import " +
                                    module_name + "\n"
                                                  "analysis = " +
                                    module_name + "." + module_name + "()",
                                globals, locals);

  _analysis = locals["analysis"];

  if (py::hasattr(_analysis, "StartAnalysis"))
    pySafeCall(_analysis.attr("StartAnalysis"), analysis_args);
}

void OperatorObserver::startNetworkExecution(loco::Graph *graph)
{
  if (!py::hasattr(_analysis, "StartNetworkExecution"))
    return;

  const auto input_nodes = loco::input_nodes(graph);
  py::list inputs;
  // Assumption: input_nodes is iterated in the same order of model inputs
  for (const auto input_node : input_nodes)
  {
    auto circle_node = loco::must_cast<luci::CircleInput *>(input_node);
    inputs.append(outputPyArray(circle_node, _interpreter));
  }
  pySafeCall(_analysis.attr("StartNetworkExecution"), inputs);
}

void OperatorObserver::endNetworkExecution(loco::Graph *graph)
{
  if (!py::hasattr(_analysis, "EndNetworkExecution"))
    return;

  const auto output_nodes = loco::output_nodes(graph);
  py::list outputs;
  // Assumption: output_nodes is iterated in the same order of model outputs
  for (const auto output_node : output_nodes)
  {
    auto circle_node = loco::must_cast<luci::CircleOutput *>(output_node);
    outputs.append(
        outputPyArray(loco::must_cast<luci::CircleNode *>(circle_node->from()), _interpreter));
  }
  pySafeCall(_analysis.attr("EndNetworkExecution"), outputs);
}

void OperatorObserver::endAnalysis()
{
  if (py::hasattr(_analysis, "EndAnalysis"))
    pySafeCall(_analysis.attr("EndAnalysis"));
}

} // namespace dalgona
