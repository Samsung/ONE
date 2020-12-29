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

#include <algorithm>
#include <sstream>

#include "Linear.h"

#include "backend/IConfig.h"
#include "backend/Backend.h"
#include "util/logging.h"
#include "dumper/text/GraphDumper.h"

namespace onert
{
namespace compiler
{

std::vector<ir::OpSequenceIndex> Linear::linearize(const compiler::LoweredGraph &lowered_graph)
{
  std::vector<ir::OpSequenceIndex> order;
  lowered_graph.iterateTopolOpSeqs(
    [&](const ir::OpSequenceIndex &index, const ir::OpSequence &) -> void {
      order.emplace_back(index);
    });
  return order;
}

void Linear::dump(const compiler::LoweredGraph &lowered_graph,
                  const std::vector<ir::OpSequenceIndex> &order)
{
  {
    VERBOSE(Linear) << "Final OpSequences" << std::endl;
    for (const auto index : order)
    {
      // TODO Could logging system can handle this? (Inserting prefix for each line)
      std::istringstream iss{dumper::text::formatOpSequence(lowered_graph, index)};
      std::string line;
      while (std::getline(iss, line))
        VERBOSE(GraphDumper) << line << std::endl;
    }
  }
}

} // namespace compiler
} // namespace onert
