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

#include "PartitionPModules.h"
#include "ConnectNode.h"

#include "luci/Service/CircleNodeClone.h"
#include "luci/Log.h"

#include <loco.h>

namespace
{

// forward declare
void clone_ifnode_subgraphs(luci::PartedModule &pm, const luci::CircleIf *if_node,
                            const luci::CloneContext &clonectx);

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

/**
 * @brief make a clone of graph
 */
std::unique_ptr<loco::Graph> clone_graph(loco::Graph *graph_org, luci::CloneContext &clonectx)
{
  auto graph = loco::make_graph();
  auto graph_clone = graph.get();

  graph_clone->name(graph_org->name());

  // clone inputs
  for (uint32_t n = 0; n < graph_org->inputs()->size(); ++n)
  {
    auto input_org = luci::input_node(graph_org, n);
    assert(input_org != nullptr);

    auto *input_clone = graph_clone->nodes()->create<luci::CircleInput>();
    luci::copy_common_attributes(input_org, input_clone);

    add_graph_input(graph_clone, input_clone);
    clonectx.emplace(input_org, input_clone);
  }

  // clone nodes
  auto nodes = graph_org->nodes();
  for (uint32_t n = 0; n < nodes->size(); ++n)
  {
    auto node = nodes->at(n);

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
  for (uint32_t n = 0; n < nodes->size(); ++n)
  {
    auto node = nodes->at(n);

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

  return std::move(graph);
}

void clone_recursive_subgraphs(luci::PartedModule &pm, loco::Graph *graph,
                               const luci::CloneContext &clonectx)
{
  auto nodes = graph->nodes();
  for (uint32_t n = 0; n < nodes->size(); ++n)
  {
    auto if_node = dynamic_cast<luci::CircleIf *>(nodes->at(n));
    if (if_node != nullptr)
    {
      clone_ifnode_subgraphs(pm, if_node, clonectx);
    }
    // TODO handle While
  }
}

void clone_ifnode_subgraphs(luci::PartedModule &pm, const luci::CircleIf *if_node,
                            const luci::CloneContext &clonectx)
{
  assert(if_node != nullptr);

  auto it = clonectx.find(if_node);
  assert(it != clonectx.end());
  auto if_clone = loco::must_cast<luci::CircleIf *>(it->second);

  luci::CloneContext then_clonectx;
  luci::CloneContext else_clonectx;

  auto then_graph = if_node->then_graph();
  auto else_graph = if_node->else_graph();

  auto then_clone = clone_graph(then_graph, then_clonectx);
  auto else_clone = clone_graph(else_graph, else_clonectx);
  if_clone->then_graph(then_clone.get());
  if_clone->else_graph(else_clone.get());

  pm.module->add(std::move(then_clone));
  int32_t then_index = pm.module->size() - 1;
  pm.module->add(std::move(else_clone));
  int32_t else_index = pm.module->size() - 1;
  if_clone->then_branch(then_index);
  if_clone->else_branch(else_index);

  // do recursive copy subgraphs of CircleIf if there are any,
  // inside then_graph or else_graph.
  clone_recursive_subgraphs(pm, then_graph, then_clonectx);
  clone_recursive_subgraphs(pm, else_graph, else_clonectx);
}

/**
 * @brief Build loco::graph from pgroup into graph
 */
void build_graph(luci::PartedModule &pm, loco::Graph *graph, const luci::PGroup *pgroup)
{
  LOGGER(l);

  luci::CloneContext clonectx;

  // add input node(s)
  for (auto *input : pgroup->inputs)
  {
    auto *input_clone = graph->nodes()->create<luci::CircleInput>();
    luci::copy_common_attributes(input, input_clone);

    add_graph_input(graph, input_clone);
    clonectx.emplace(input, input_clone);

    INFO(l) << "MAP: "
            << " input(" << input << ") -> " << input_clone << "(" << input_clone->name() << ")";
  }

  // add CircleConst for inputs
  for (auto &pnode : pgroup->pnodes)
  {
    auto node = pnode->node;
    uint32_t arity = node->arity();
    for (uint32_t a = 0; a < arity; ++a)
    {
      auto in_a_const = dynamic_cast<luci::CircleConst *>(node->arg(a));
      if (in_a_const != nullptr)
      {
        auto it = clonectx.find(in_a_const);
        if (it == clonectx.end())
        {
          auto *clone = clone_node(in_a_const, graph);
          clonectx.emplace(in_a_const, clone);

          INFO(l) << "MAP: "
                  << " const(" << in_a_const << ") -> " << clone << "(" << clone->name() << ")";
        }
      }
    }
  }

  // add nodes
  for (auto &pnode : pgroup->pnodes)
  {
    auto *clone = clone_node(pnode->node, graph);
    clonectx.emplace(pnode->node, clone);

    INFO(l) << "MAP: "
            << "  node(" << pnode->node << ") -> " << clone << "(" << clone->name() << ")";
  }
  // connect nodes
  for (auto &pnode : pgroup->pnodes)
  {
    clone_connect(pnode->node, clonectx);
  }

  // add output node(s)
  for (auto *output : pgroup->outputs)
  {
    auto *output_clone = graph->nodes()->create<luci::CircleOutput>();
    luci::copy_common_attributes(output, output_clone);
    // note: we don't add output_clone to clonectx.
    // logically, output is not used as an input to any other nodes.

    auto it = clonectx.find(output);
    assert(it != clonectx.end());
    output_clone->from(it->second);

    add_graph_output(graph, output_clone);

    INFO(l) << "MAP: "
            << "output(" << output << ") -> " << output_clone << "(" << output_clone->name() << ")"
            << ": from " << it->second << "(" << it->second->name() << ")";
  }

  // TODO relocate this if needed
  // subgraphs for IF/WHILE/... nodes
  for (auto &pnode : pgroup->pnodes)
  {
    auto if_node = dynamic_cast<const luci::CircleIf *>(pnode->node);
    if (if_node != nullptr)
    {
      clone_ifnode_subgraphs(pm, if_node, clonectx);
    }
    // TODO handle While
  }
}

std::string make_name(const luci::PGroup *pgroup)
{
  auto &first_pnode = *pgroup->pnodes.begin();
  auto *first_node = first_pnode->node;
  std::string name = first_node->graph()->name();
  name = name + "_" + pgroup->group;
  return name;
}

} // namespace

namespace luci
{

/**
 * @brief This will produce list of luci::Module as PartedModules from pgroups
 */
luci::PartedModules produce_pmodules(const luci::PGroups *pgroups)
{
  LOGGER(l);

  luci::PartedModules pms;

  for (auto &pgroup : pgroups->pgroups)
  {
    luci::PartedModule pm;
    pm.module = std::make_unique<luci::Module>();
    pm.group = pgroup->group;

    // the main graph for this module
    auto graph = loco::make_graph();
    auto graph_ptr = graph.get();

    auto graph_name = make_name(pgroup.get());
    graph->name(graph_name);

    // Add main graph so that other subgraphs can be added inside build_graph
    pm.module->add(std::move(graph));

    INFO(l) << "--- Partition Graph build----------------------";
    INFO(l) << "--- name: " << graph_name;
    build_graph(pm, graph_ptr, pgroup.get());

    pms.pmodules.emplace_back(std::move(pm));
  }

  return pms;
}

} // namespace luci
