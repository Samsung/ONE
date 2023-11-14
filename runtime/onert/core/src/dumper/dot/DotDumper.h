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
#include "ir/train/TrainableGraph.h"
#include "compiler/ILoweredGraph.h"

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
  DotDumper(Level level) : _level{level} {}

public:
  /**
   * @brief Dump graph information to dot file as tag name if "GRAPH_DOT_DUMP" is set
   *
   * @param[in] graph  The graph that would be used to get operations and operands
   * @param[in] tag    The name of dot file that would be created
   * @return N/A
   */
  void dump(const ir::Graph &graph, const std::string &tag);

  /**
   * @brief Dump lowered graph information to dot file as tag name if "GRAPH_DOT_DUMP" is set
   *
   * @param[in] graph  The graph that would be used to get operations and operands
   * @param[in] tag    The name of dot file that would be created
   * @return N/A
   */
  void dump(const compiler::ILoweredGraph &lowered_graph, const std::string &tag);

  /**
   * @brief Dump graph information to dot file as tag name if "GRAPH_DOT_DUMP" is set
   *
   * @param[in] graph  TrainableGraph to be dumped
   * @param[in] tag    The name of dot file to be dumped
   * @return N/A
   */
  void dump(const ir::train::TrainableGraph &graph, const std::string &tag);

private:
  Level _level;
};

} // namespace dot
} // namespace dumper
} // namespace onert

#endif // __ONERT_DUMPER_DOT_DOT_DUMPER_H__
