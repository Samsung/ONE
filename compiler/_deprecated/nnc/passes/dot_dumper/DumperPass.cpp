/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "passes/dot_dumper/DumperPass.h"
#include "mir/Graph.h"
#include "mir/IrDotDumper.h"

#include <fstream>

namespace nnc
{

using namespace mir;
int DumperPass::_counter = 0;

PassData DumperPass::run(PassData data)
{
  auto graph = static_cast<Graph *>(data);
  assert(graph && "graph object is expected");
  std::ofstream stream(std::to_string(_counter++) + "_" + _file_name + ".dot");
  dumpGraph(graph, stream);
  return graph;
}

} // namespace nnc
