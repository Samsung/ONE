/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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
#include "ModuleCloner.h"

#include <luci/ConnectNode.h>
#include <luci/Service/CircleNodeClone.h>

using namespace mpqsolver;

namespace
{

void add_graph_input(loco::Graph *graph, luci::CircleInput *input_node)
{
  assert(graph != nullptr);
  assert(input_node != nullptr);

  auto graph_input = graph->inputs()->create();
  graph_input->name(input_node->name());

  // Set GraphInputOutputIndex for graph
  input_node->index(graph_input->index());

  // Data type
  graph_input->dtype(input_node->dtype());

  // Shape of GraphInput
  auto input_shape = std::make_unique<loco::TensorShape>();
  input_shape->rank(input_node->rank());
  for (uint32_t r = 0; r < input_node->rank(); ++r)
  {
    if (input_node->dim(r).known())
      input_shape->dim(r).set(input_node->dim(r).value());
  }
  graph_input->shape(std::move(input_shape));
}

void add_graph_output(loco::Graph *graph, luci::CircleOutput *output_node)
{
  assert(graph != nullptr);
  assert(output_node != nullptr);

  auto graph_output = graph->outputs()->create();
  graph_output->name(output_node->name());

  // Set GraphInputOutputIndex for graph
  output_node->index(graph_output->index());

  // Data type
  graph_output->dtype(output_node->dtype());

  // Shape of GraphOutput
  auto output_shape = std::make_unique<loco::TensorShape>();
  output_shape->rank(output_node->rank());
  for (uint32_t r = 0; r < output_node->rank(); ++r)
  {
    if (output_node->dim(r).known())
      output_shape->dim(r).set(output_node->dim(r).value());
  }
  graph_output->shape(std::move(output_shape));
}

std::unique_ptr<loco::Graph> clone_graph(loco::Graph *graph_org)
{
  if (not graph_org)
    return nullptr;
  luci::CloneContext clonectx;

  auto graph = loco::make_graph();
  auto graph_clone = graph.get();
  auto &graph_name = graph_org->name();

  graph_clone->name(graph_name);

  // clone inputs
  auto inputs = graph_org->inputs();
  assert(inputs);
  for (const auto &node : loco::input_nodes(graph_org))
  {
    auto input_node = loco::must_cast<const luci::CircleNode *>(node);

    auto *input_clone = graph_clone->nodes()->create<luci::CircleInput>();
    luci::copy_common_attributes(input_node, input_clone);

    add_graph_input(graph_clone, input_clone);
    clonectx.emplace(input_node, input_clone);
  }

  // clone nodes
  auto nodes = loco::all_nodes(graph_org);
  for (const auto &node : nodes)
  {
    auto cnode = loco::must_cast<const luci::CircleNode *>(node);

    // skip for CircleInput, CircleOutput
    if (dynamic_cast<luci::CircleInput *>(node) != nullptr)
      continue;
    if (dynamic_cast<luci::CircleOutput *>(node) != nullptr)
      continue;

    auto node_org = loco::must_cast<luci::CircleNode *>(node);
    assert(clonectx.find(node_org) == clonectx.end());

    auto *node_clone = clone_node(node_org, graph_clone);
    clonectx.emplace(node_org, node_clone);
  }

  // connect nodes
  for (auto node : nodes)
  {
    // skip for CircleInput, CircleOutput
    if (dynamic_cast<luci::CircleInput *>(node) != nullptr)
      continue;
    if (dynamic_cast<luci::CircleOutput *>(node) != nullptr)
      continue;

    auto node_org = loco::must_cast<luci::CircleNode *>(node);
    clone_connect(node_org, clonectx);
  }

  // clone outputs
  for (uint32_t n = 0; n < graph_org->outputs()->size(); ++n)
  {
    auto output_org = luci::output_node(graph_org, n);
    assert(output_org != nullptr);

    auto *output_clone = graph_clone->nodes()->create<luci::CircleOutput>();
    luci::copy_common_attributes(output_org, output_clone);
    // note: we don't add output_clone to clonectx.
    // logically, output is not used as an input to any other nodes.
    auto output_from = loco::must_cast<luci::CircleNode *>(output_org->from());
    auto it = clonectx.find(output_from);
    assert(it != clonectx.end());
    output_clone->from(it->second);

    add_graph_output(graph_clone, output_clone);
  }

  return graph;
}

} // namespace

std::unique_ptr<luci::Module> ModuleCloner::clone(const luci::Module *module)
{
  std::unique_ptr<luci::Module> new_module = luci::make_module();
  for (size_t idx = 0; idx < module->size(); ++idx)
  {
    auto graph = module->graph(idx);
    if (!graph)
    {
      return nullptr;
    }
    auto new_graph = clone_graph(graph);
    new_module->add(std::move(new_graph));
  }

  return new_module;
}
