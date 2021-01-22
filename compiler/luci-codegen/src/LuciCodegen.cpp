/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "LuciCodegen.h"
#include "CodegenKernelBuilder.h"
#include "SubgraphContext.h"
#include "Utilities.h"

#include "loco/IR/Algorithm.h"

#include "Halide.h"

#include <map>

namespace luci_codegen
{

LuciCodegen::LuciCodegen(const Options &options) : _options(options) {}

LuciCodegen::~LuciCodegen() {}

bool LuciCodegen::fits_constrains(luci::CircleNode *node)
{
  if (node->opcode() == luci::CircleOpcode::CIRCLECONST)
    return const_node_size(node) <= _options.max_inline_buffer_threshold;
  return is_supported(node);
}

void LuciCodegen::add_operator(luci::CircleNode *node, SubgraphContext &subgraph)
{
  assert(fits_constrains(node));
  CodegenKernelBuilder builder(subgraph);
  node->accept(&builder);
}

void LuciCodegen::process_graph(loco::Graph &graph)
{
  SubgraphContext subgraph;
  auto *inputs = graph.inputs();
  auto input = inputs->at(0);
  auto outputs = loco::output_nodes(&graph);
  for (loco::Node *node: loco::postorder_traversal(outputs))
  {
    auto circle_node = static_cast<luci::CircleNode *>(node);
    if (fits_constrains(circle_node))
      add_operator(circle_node, subgraph);
  }
  _compiled_subgraphs.push_back(std::move(subgraph));
}

void LuciCodegen::process_module(luci::Module &module)
{
  auto num_graphs = module.size();
  for (size_t i = 0; i < num_graphs; ++i)
    process_graph(*module.graph(i));
}

void LuciCodegen::emit_code(std::string package_name)
{
  int no = 0;
  for (auto subgraph: _compiled_subgraphs)
    for (auto &node_func: subgraph.generated_funcs())
    {
      ++no;
      node_func.second.compile_to_lowered_stmt("func_" + std::to_string(no) + ".html", subgraph.inputs(), Halide::StmtOutputFormat::HTML);
    }
  // TODO generate object files?
}

} // namespace luci_codegen
