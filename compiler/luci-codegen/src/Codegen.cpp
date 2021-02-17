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

#include "Codegen.h"
#include "KernelBuilder.h"
#include "SubgraphContext.h"
#include "Scheduler.h"
#include "Utilities.h"

#include "luci/IR/Nodes/CircleCustom.h"
#include "luci/IR/Nodes/CircleCustomOut.h"
#include "loco/IR/Algorithm.h"

#include "Halide.h"

#include "flatbuffers/flexbuffers.h"

#include <map>
#include <unordered_set>
#include <algorithm>

namespace
{

std::vector<uint8_t> create_custom_options(const std::string &name)
{
  flexbuffers::Builder fbb;
  fbb.Map([&]() {fbb.String("func_name", name);});
  fbb.Finish();
  return fbb.GetBuffer();
}

} // unnamed namespace

namespace luci_codegen
{

Codegen::Codegen(const CodegenOptions &options) : _processed_graphs(0), _options(options) {}

Codegen::~Codegen() {}

bool Codegen::fits_constrains(luci::CircleNode *node) const
{
  if (node->opcode() == luci::CircleOpcode::CIRCLECONST)
    return const_node_size(node) <= _options.max_inline_buffer_threshold;
  return KernelBuilder::is_supported(node);
}

std::vector<luci::CircleNode *>
Codegen::gather_suitable_nodes(luci::CircleNode *node)
{
  std::vector<luci::CircleNode *> subgraph_nodes;
  std::queue<luci::CircleNode *> queue;
  queue.push(node);
  _processed.insert(node);
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
      if (_processed.count(adj) || !fits_constrains(adj))
      {
        continue;
      }
      _processed.insert(adj);
      queue.push(adj);
    }
  }
  return subgraph_nodes;
}

/**
 * This function checks if there are no forbidden paths through graphs,
 * so after replacement of compiled subgraph generated node will be dependent from itself
 * @param nodes
 * @return
 */
bool has_self_dependency_subgraph(std::vector<luci::CircleNode *> nodes)
{
  std::unordered_set<loco::Node *> belong_to_subgraph;
  belong_to_subgraph.insert(nodes.begin(), nodes.end());
  // gather input nodes
  std::unordered_set<loco::Node *> inputs;
  for (auto *node: nodes)
  {
    for (int i = 0; i < node->arity(); ++i)
    {
      loco::Node *prev = node->arg(i);
      // We do not care if same input will be inserted several times, so no checks for this
      // only first appearance will take effect
      if (belong_to_subgraph.count(prev) == 0)
      {
        inputs.insert(prev);
      }
    }
  }
  // gather successors of input nodes and constants belonging subgraph
  std::queue<loco::Node *> queue;
  std::unordered_set<loco::Node *> visited;
  for (loco::Node *node: nodes)
  {
    if (static_cast<luci::CircleNode *>(node)->opcode() == luci::CircleOpcode::CIRCLECONST)
    {
      queue.push(node);
      visited.insert(node);
    }
  }
  for (loco::Node *input: inputs)
  {
    for (loco::Node *succ: loco::succs(input))
    {
      if (belong_to_subgraph.count(succ) != 0)
      {
        queue.push(succ);
        visited.insert(succ);
      }
    }
  }
  // check reachability of inputs from start_nodes
  // traverse nodes, to get all intermediate nodes
  while (!queue.empty())
  {
    loco::Node *node = queue.front();
    queue.pop();
    for (loco::Node *succ: loco::succs(node))
    {
      if (visited.count(succ) != 0)
      {
        continue;
      }
      visited.insert(succ);
      if (inputs.count(succ) != 0)
      {
        // this means algorithm reached subgraph input from nodes of this subgraph, we found cyclic dependency
        return true;
      }
      queue.push(succ);
    }
  }
  return false;
}

// check if we can compile found subgraph and remove redundant nodes
// Example of problematic subgraph:
// C - compilable node
// N - not compilable node
//
//   |
//   C
//  / \
// N  C
//  \ /
//   C
//   |
//
// this graph will be transformed into graph with cyclic dependency of generated node from itself
std::vector<std::vector<luci::CircleNode *>>
Codegen::extract_subgraphs(const std::vector<luci::CircleNode *> &nodes) const
{
  if (has_self_dependency_subgraph(nodes))
    return {};
  return {nodes};
}

SubgraphContext *Codegen::create_subgraph(const std::vector<luci::CircleNode *> &nodes)
{
  std::string subgraph_name = "generated_subgraph_" + std::to_string(_processed_graphs);
  _compiled_subgraphs.emplace_back(subgraph_name, std::move(nodes));
  auto *subgraph = &_compiled_subgraphs.back();
  subgraph->finish_nodes_construction();
  return subgraph;
}

void Codegen::replace_subgraph_with_generated_node(SubgraphContext *subgraph) const
{
  auto &inputs = subgraph->get_inputs();
  const auto num_inputs = inputs.size();
  loco::Graph *graph = subgraph->get_graph();

  auto compiled_node = graph->nodes()->create<luci::CircleCustom>(num_inputs);
  compiled_node->custom_code("COMPILED_OP");

  auto options = create_custom_options(subgraph->get_name());
  compiled_node->custom_options(options);

  compiled_node->dtype(loco::DataType::FLOAT32);

  for (int i = 0; i < num_inputs; ++i)
  {
    compiled_node->inputs(i, subgraph->get_inputs()[i].first);
  }

  for (int i = 0; i < subgraph->get_outputs().size(); ++i)
  {
    auto output = subgraph->get_outputs()[i];
    auto custom_output = graph->nodes()->create<luci::CircleCustomOut>();
    custom_output->input(compiled_node);
    custom_output->index(i);
    custom_output->dtype(output.first->dtype());
    custom_output->shape_status(output.first->shape_status());

    // copy shape
    uint32_t rank = output.first->rank();
    custom_output->rank(rank);
    for (uint32_t i = 0; i < rank; ++i)
    {
      custom_output->dim(i) = output.first->dim(i);
    }

    loco::replace(output.first).with(custom_output);
  }
}

void Codegen::cleanup_graph(SubgraphContext *subgraph) const
{
  loco::Graph *graph = subgraph->get_graph();
  std::vector<loco::Node *> outputs;
  for (auto node: subgraph->get_outputs())
  {
    outputs.push_back(node.first);
  }
  auto ordered_nodes = loco::postorder_traversal(outputs);
  std::reverse(ordered_nodes.begin(), ordered_nodes.end());
  for (auto node: ordered_nodes)
  {
    if (subgraph->contains(static_cast<luci::CircleNode *>(node)))
      graph->nodes()->destroy(node);
  }
}

Halide::Target get_halide_target(const CodegenOptions &options)
{
  Halide::Target target = Halide::get_host_target();
  target.set_features({});

  if (!options.debug)
    target.set_feature(Halide::Target::NoAsserts);

  switch (options.os)
  {
    case OS::Linux:
      target.os = Halide::Target::Linux;
      break;
    case OS::Windows:
      target.os = Halide::Target::Windows;
      break;
    case OS::Android:
      target.os = Halide::Target::Android;
      break;
    case OS::Native:
      // Do nothing
      break;
    default:
      assert(false && "unsupported OS");
      break;
  }

  switch (options.arch.type)
  {
    case ArchType::ARM_32:
      target.arch = Halide::Target::ARM;
      target.bits = 32;
      break;
    case ArchType::ARM_64:
      target.arch = Halide::Target::ARM;
      target.bits = 64;
      break;
    case ArchType::X86_32:
      target.arch = Halide::Target::X86;
      target.bits = 32;
      break;
    case ArchType::X86_64:
      target.arch = Halide::Target::X86;
      target.bits = 64;
      break;
    case ArchType::Native:
      // Do nothing
      break;
    default:
      assert(false && "unsupported arch");
      break;
  }
  return target;
}

void Codegen::process_graph(loco::Graph &graph)
{
  auto nodes = graph.nodes();

  Halide::Target target = get_halide_target(_options);

  // find and generate code
  for (int i = 0; i < nodes->size(); ++i)
  {
    auto node = static_cast<luci::CircleNode *>(nodes->at(i));

    // Check if we found node that belongs to subgraph we can compile
    if (_processed.count(node) || !fits_constrains(node))
      continue;

    // Traverse graph to find all compilable adjacent nodes
    std::vector<luci::CircleNode *> suitable_nodes = gather_suitable_nodes(node);

    std::vector<std::vector<luci::CircleNode *>> extracted_subgraph_nodes = extract_subgraphs(suitable_nodes);

    for (auto &nodes: extracted_subgraph_nodes)
    {
      SubgraphContext *subgraph = create_subgraph(nodes);

      subgraph->set_target(target);

      // Create kernels for nodes
      KernelBuilder(*subgraph).process();

      SchedulerOptions scheduler_options = {_options.scheduler, _options.arch.l1_size};
      Scheduler(*subgraph, scheduler_options).process();

      _processed_graphs++;
    }
  }

  // replace circle graph with generated nodes
  for (SubgraphContext &subgraph: _compiled_subgraphs)
  {
    // Replace subgraph with custom operator
    replace_subgraph_with_generated_node(&subgraph);

    // Cleanup graph
    cleanup_graph(&subgraph);
  }
}

void Codegen::process_module(luci::Module &module)
{
  auto num_graphs = module.size();
  for (size_t i = 0; i < num_graphs; ++i)
    process_graph(*module.graph(i));
}

void Codegen::emit_code(std::string package_name)
{
  for (auto &subgraph: _compiled_subgraphs)
  {
    Halide::Pipeline &pipeline = subgraph.get_pipeline();
    Halide::Target target = subgraph.get_target();

    std::vector<Halide::Argument> arguments;
    for (auto input: subgraph.get_inputs())
    {
      arguments.push_back(input.second);
    }

    Halide::Module module = pipeline.compile_to_module(arguments, subgraph.get_name(), target, Halide::LinkageType::ExternalPlusMetadata);
    module.set_auto_scheduler_results(subgraph.get_schedule());

    std::map<Halide::Output, std::string> products;

    products[Halide::Output::object] = subgraph.get_name() + ".o";

    module.compile(products);
  }
}

} // namespace luci_codegen
