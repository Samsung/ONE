/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "mir/IrDotDumper.h"
#include "mir/Graph.h"
#include "DotGraph.h"
#include "DotNodeBuilder.h"

namespace mir
{

void dumpGraph(const Graph *graph, std::ostream &stream)
{
  DotGraph dot_graph;

  for (const auto *node : graph->getNodes())
  {
    dot_graph.addNode(DotNodeBuilder(*node).getDotNode());
    for (const Operation::Output *input : node->getInputs())
    {
      dot_graph.addEdge({input->getNode()->getId(), node->getId()});
    }
  }

  stream << dot_graph;
}

} // namespace mir
