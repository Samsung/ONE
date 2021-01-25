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
#include <unordered_set>

namespace luci_codegen
{

LuciCodegen::LuciCodegen(const Options &options) : _options(options) {}

LuciCodegen::~LuciCodegen() {}

bool LuciCodegen::fits_constrains(luci::CircleNode *node) const
{
  if (node->opcode() == luci::CircleOpcode::CIRCLECONST)
    return const_node_size(node) <= _options.max_inline_buffer_threshold;
  return is_supported(node);
}

std::vector<luci::CircleNode *> LuciCodegen::gather_suitable_nodes(luci::CircleNode *node, std::unordered_set<luci::CircleNode *> &processed) const
{
  std::vector<luci::CircleNode *> subgraph_nodes;
  std::queue<luci::CircleNode *> queue;
  queue.push(node);
  processed.insert(node);
  while (!queue.empty())
  {
    luci::CircleNode *cur_node = queue.front();
    subgraph_nodes.push_back(cur_node);
    queue.pop();

    std::vector<luci::CircleNode *> adjacent;
    // gather adjacent nodes
    for (int i = 0; i < cur_node->arity(); ++i)
    {
      adjacent.push_back(static_cast<luci::CircleNode *>(cur_node->arg(i)));
    }
    auto succs = loco::succs(cur_node);
    for (auto succ: succs)
    {
      adjacent.push_back(static_cast<luci::CircleNode *>(succ));
    }
    // process adjacent nodes
    for (auto adj: adjacent)
    {
      if (processed.count(adj) || !fits_constrains(adj))
      {
        continue;
      }
      processed.insert(adj);
      queue.push(adj);
    }
  }
  return subgraph_nodes;
}

void LuciCodegen::process_graph(loco::Graph &graph)
{
  std::unordered_set<luci::CircleNode *> processed;
  auto nodes = graph.nodes();

  for (int i = 0; i < nodes->size(); ++i)
  {
    auto node = static_cast<luci::CircleNode *>(nodes->at(i));
    if (processed.count(node) || !fits_constrains(node))
      continue;

    // Traverse graph to find adjacent supported nodes
    std::vector<luci::CircleNode *> subgraph_nodes = gather_suitable_nodes(node, processed);

    _compiled_subgraphs.emplace_back(std::move(subgraph_nodes));
    auto &subgraph = _compiled_subgraphs.back();
    subgraph.finish_construction();

    CodegenKernelBuilder kernel_builder(subgraph);

    // TODO make separate scheduler entity for this
    for (auto node: subgraph.get_nodes())
      kernel_builder.visit(node);

    // Replace subgraph with custom operator
    auto &inputs = subgraph.inputs();
    const auto num_inputs = inputs.size();

    auto compiled_node = graph.nodes()->create<luci::CircleCustom>(num_inputs);
    compiled_node->custom_code("COMPILED_OP");

    for (int i = 0; i < num_inputs; ++i)
    {
      compiled_node->inputs(i, subgraph.inputs()[i].first);
    }

    for (int i = 0; i < subgraph.outputs().size(); ++i)
    {
      auto output = subgraph.outputs()[i];
      auto custom_output = graph.nodes()->create<luci::CircleCustomOut>();
      custom_output->input(compiled_node);
      custom_output->index(i);
      loco::replace(output.first).with(custom_output);
    }

    // Cleanup graph
    for (auto node: subgraph.get_nodes())
    {
      graph.nodes()->destroy(node);
    }
  }
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
  for (auto &subgraph: _compiled_subgraphs)
  {
    std::vector<Halide::Argument> arguments;
    for (auto input: subgraph.inputs())
    {
      arguments.push_back(input.second);
    }
    std::vector<Halide::Func> outputs;
    for (auto output: subgraph.outputs())
    {
      outputs.push_back(output.second);
    }
    Halide::Pipeline composite_output(outputs);
    ++no;
    composite_output.compile_to_lowered_stmt("func_" + std::to_string(no) + ".html", arguments, Halide::StmtOutputFormat::HTML);
  }
  // TODO generate object files/static libraries?
}

} // namespace luci_codegen
