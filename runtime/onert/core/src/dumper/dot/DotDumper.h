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

#include "ir/Graph.h"
#include "compiler/LoweredGraph.h"

#ifndef __ONERT_DUMPER_DOT_DOT_DUMPER_H__
#define __ONERT_DUMPER_DOT_DOT_DUMPER_H__

namespace onert
{
namespace dumper
{
namespace dot
{

class DotDumper
{
public:
  enum Level
  {
    OFF = 0,               //< Do not dump
    ALL_BUT_CONSTANTS = 1, //< Emit all operations and operands but constants
    ALL = 2                //< Emit all operations and operands
  };

public:
  DotDumper(const ir::Graph &graph, Level level)
      : _lowered_graph{nullptr}, _graph(graph), _level{level}
  {
  }
  DotDumper(const compiler::LoweredGraph *lowered_graph, Level level)
      : _lowered_graph{lowered_graph}, _graph(_lowered_graph->graph()), _level{level}
  {
  }

public:
  /**
   * @brief Dump to dot file as tag name if "GRAPH_DOT_DUMP" is set
   *
   * @param[in] tag    The name of dot file that would be created
   * @return N/A
   */
  void dump(const std::string &tag);

private:
  const compiler::LoweredGraph *_lowered_graph;
  const ir::Graph &_graph;
  Level _level;
};

} // namespace dot
} // namespace dumper
} // namespace onert

#endif // __ONERT_DUMPER_DOT_DOT_DUMPER_H__
