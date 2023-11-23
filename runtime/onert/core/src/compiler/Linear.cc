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

#include "Linear.h"

#include "../dumper/text/GraphDumper.h"

#include "util/logging.h"

#include <sstream>

namespace onert
{
namespace compiler
{

// TODO(easy) Change the LoweredGraph param to Graph
std::vector<ir::OperationIndex> Linear::linearize(const compiler::ILoweredGraph &lowered_graph)
{
  return lowered_graph.graph().topolSortOperations();
}

// TODO(easy) Change the LoweredGraph param to Graph
std::vector<ir::OperationIndex> Linear::blinearize(const compiler::ILoweredGraph &lowered_graph)
{
  return lowered_graph.graph().btopolSortOperations();
}

// TODO(easy) Change the LoweredGraph param to Graph
void Linear::dump(const compiler::ILoweredGraph &lowered_graph,
                  const std::vector<ir::OperationIndex> &order)
{
  for (const auto &ind : order)
  {
    // TODO Could logging system can handle this? (Inserting prefix for each line)
    std::istringstream iss{dumper::text::formatOperation(lowered_graph.graph(), ind)};
    std::string line;
    while (std::getline(iss, line))
      VERBOSE(GraphDumper) << line << std::endl;
  }
}

} // namespace compiler
} // namespace onert
