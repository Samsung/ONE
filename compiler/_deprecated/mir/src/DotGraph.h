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

#ifndef _MIR_DOT_GRAPH_
#define _MIR_DOT_GRAPH_

#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

namespace mir
{

struct DotNode
{
  std::size_t id;
  std::string label;
};

struct DotEdge
{
  std::size_t src_id;
  std::size_t dst_id;
};

class DotGraph
{
public:
  void addNode(DotNode node);
  void addEdge(DotEdge edge);

  friend std::ostream &operator<<(std::ostream &stream, const DotGraph &graph);

private:
  std::vector<DotNode> _nodes;
  std::vector<DotEdge> _edges;
};

} // namespace mir

#endif //_MIR_DOT_GRAPH_
