/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "DotBuilder.h"

namespace onert
{
namespace dumper
{
namespace dot
{

// DotDumper
DotBuilder::DotBuilder() {}

void DotBuilder::update(const Node &node_info)
{
  add(node_info);
  for (auto edge : node_info.out_edges())
  {
    addEdge(node_info, *edge);
  }
}

void DotBuilder::writeDot(std::ostream &os)
{
  os << "digraph D {\n"
     << _dot.str() << "\n"
     << "}\n";
}

void DotBuilder::add(const Node &node)
{
  _dot << node.id();
  std::stringstream ss;
  _dot << "[";
  for (auto &attr : node.attributes())
  {
    _dot << attr.first << "=\"" << attr.second << "\" ";
  }
  _dot << "];\n";
}

void DotBuilder::addEdge(const Node &node1, const Node &node2)
{
  _dot << node1.id() << " -> " << node2.id() << ";\n";
}

} // namespace dot
} // namespace dumper
} // namespace onert
