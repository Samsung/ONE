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

#include "DotGraph.h"

namespace mir
{

void DotGraph::addNode(DotNode node) { _nodes.emplace_back(std::move(node)); }

void DotGraph::addEdge(DotEdge edge) { _edges.emplace_back(edge); }

std::ostream &operator<<(std::ostream &stream, const DotGraph &graph)
{
  stream << "digraph D {" << std::endl;
  for (const auto &node : graph._nodes)
  {
    stream << node.id << " [shape=record label=\"" << node.label << "\"];" << std::endl;
  }
  for (const auto &edge : graph._edges)
  {
    stream << edge.src_id << " -> " << edge.dst_id << ";" << std::endl;
  }
  stream << "}" << std::endl;
  return stream;
}

} // namespace mir
