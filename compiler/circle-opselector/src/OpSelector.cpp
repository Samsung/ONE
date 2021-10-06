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

#include "OpSelector.h"

#include <luci/Import/GraphBuilder.h>
#include <luci/Import/GraphBuilderContext.h>
#include <luci/Import/GraphBuilderRegistry.h>
#include <luci/Import/CircleReader.h>
#include <luci/Import/Nodes/CircleConst.h>

#include <luci/Profile/CircleNodeID.h>
#include <luci/Profile/CircleNodeOrigin.h>

#include <loco/IR/Graph.h>

#include <oops/UserExn.h>

#include <cassert>
#include <iostream>
#include <map>

#define MAIN_SUBGRAPH 0

// TODO: Re-implement to use just module.
namespace opselector
{

void OpSelector::find_unconnected_nodes(std::vector<const luci::CircleNode *> &selected_nodes,
                                        std::set<uint32_t> &used_output_tensors,
                                        std::set<uint32_t> &graph_inputs,
                                        std::set<uint32_t> &graph_outputs)
{
  const auto &operators = _reader.operators();

  std::set<uint32_t> selected_input_tensors;
  std::set<uint32_t> selected_output_tensors;

  std::vector<const circle::OperatorT *> selected_operators;

  // enroll all output nodes.
  for (auto &op : operators)
  {
    for (auto output : op.get()->outputs)
    {
      used_output_tensors.insert(output);
    }
  }

  for (auto input : _reader.inputs()) // graph's input must not have preceding node.
  {
    used_output_tensors.insert(input);
  }

  // select operators.
  for (auto cnode : selected_nodes)
  {
    uint32_t node_id = luci::get_node_id(cnode);
    selected_operators.push_back(operators[node_id].get()); // put selected nodes in vector.

    if (cnode->name().find("while") != std::string::npos ||
        cnode->name().find("if") != std::string::npos) // if has while of if node,
    {
      _has_subgraph = true; // A flag indicating whether to copy the subgraph or not,
    }
  }

  print_selected_nodes(selected_nodes);

  // add all selected tensors.
  for (auto op : selected_operators)
  {
    for (auto input : op->inputs)
    {
      selected_input_tensors.insert(input);
    }
    for (auto output : op->outputs)
    {
      selected_output_tensors.insert(output);
    }
  }
  // find and add unconnected node's output
  for (auto op : selected_operators)
  {
    bool output_connected = false;

    for (auto output : op->outputs) // check connection
    {
      if (selected_input_tensors.find(output) != selected_input_tensors.end())
      {
        output_connected = true;
      }
    }

    if (not output_connected) // if not connected other selected nodes, add all outputs
    {
      for (auto output : op->outputs)
      {
        graph_outputs.insert(output);
      }
    }
  }
  // find and add unconnected node's input
  for (auto op : selected_operators)
  {
    bool input_connected = false;

    for (auto input : op->inputs)
    {
      graph_inputs.insert(input);
      if (selected_output_tensors.find(input) != selected_output_tensors.end())
      {
        graph_inputs.erase(input);
      }
    }
  }
}

void OpSelector::print_selected_nodes(std::vector<const luci::CircleNode *> selected_nodes)
{
  const auto &operators = _reader.operators();
  const auto &tensors = _reader.tensors();

  // print selected operator's input, output nodes.
  for (auto cnode : selected_nodes)
  {
    uint32_t node_id = luci::get_node_id(cnode);
    std::string node_name = cnode->name();

    std::cout << "============== Operator[" << node_id << "] Name: " << node_name
              << " ==============" << std::endl;
    std::cout << "    <INPUT>" << std::endl;
    for (auto node : operators[node_id].get()->inputs)
    {
      std::cout << "id: " << node << " "
                << "input: " << tensors[node]->name << std::endl;
    }
    std::cout << "    <OUTPUT>" << std::endl;
    for (auto node : operators[node_id].get()->outputs)
    {
      std::cout << "id: " << node << " "
                << "output: " << tensors[node]->name << std::endl;
    }
    std::cout << std::endl;
  }
}

void OpSelector::build_cache_outputs(luci::GraphBuilderContext &gb_context)
{
  auto tensoroutputs = gb_context.tensoroutputs();

  const auto &operators = _reader.operators();

  // build a cache to identify if a tensor is output of an operator
  // if this is set, we should not create a CircleConst for this tensor
  for (uint32_t i = 0; i < operators.size(); ++i)
  {
    const circle::OperatorT &op = *operators[i];
    const auto &outputs = op.outputs;

    for (uint32_t j = 0; j < outputs.size(); ++j)
    {
      auto tidx = outputs[j];
      tensoroutputs->enroll(tidx);
    }
  }
}

void OpSelector::create_graph_inputs(luci::GraphBuilderContext &gb_context, uint32_t input)
{
  auto graph = gb_context.graph();
  auto nodefinder = gb_context.nodefinder();
  auto tensoroutputs = gb_context.tensoroutputs();

  const auto &tensors = _reader.tensors();
  auto tensors_ptr = _reader.tensors_ptr();
  assert(tensors_ptr != nullptr);

  // graph inputs;
  // creating virtual input nodes will make possible to connect nodes that uses them
  // all attributes of tensor should be copied to CircleInput node
  auto input_node = graph->nodes()->create<luci::CircleInput>();

  assert(input_node != nullptr);
  const circle::TensorT &tensor = *tensors[input];

  luci::copy_tensor_attributes(tensor, input_node);

  if (tensors_ptr->Get(input)->shape() == nullptr)
  {
    input_node->shape_status(luci::ShapeStatus::NOSHAPE);
  }
  else
  {
    input_node->shape_status(luci::ShapeStatus::VALID);
  }

  nodefinder->enroll(input, input_node);

  // input_node is also an output to a tensor
  tensoroutputs->enroll(input);

  // Name
  auto graph_input = graph->inputs()->create();
  graph_input->name(input_node->name());

  // Set GraphInputOutputIndex for graph
  input_node->index(graph_input->index());

  // Data type
  graph_input->dtype(input_node->dtype());

  assert(tensor.shape_signature.size() == 0 ||
         tensor.shape_signature.size() == tensor.shape.size());

  // Shape of GraphInput
  auto input_shape = std::make_unique<loco::TensorShape>();
  const std::vector<int32_t> &input_dims = tensor.shape; // in NHWC
  input_shape->rank(input_dims.size());
  for (uint32_t r = 0; r < input_dims.size(); ++r)
  {
    if (tensor.shape_signature.size() > 0 && tensor.shape_signature.at(r) == -1)
    {
      input_shape->dim(r).unset();
    }
    else
    {
      input_shape->dim(r).set(input_dims[r]);
    }
  }
  graph_input->shape(std::move(input_shape));
}

void OpSelector::create_circle_const(luci::GraphBuilderContext &gb_context)
{
  auto nodefinder = gb_context.nodefinder();

  const auto &tensors = _reader.tensors();

  // Create CircleConst nodes for constant tensors.
  for (uint32_t i = 0; i < tensors.size(); ++i)
  {
    luci::CircleConst *const_node = luci::create_circleconst(&gb_context, i);
    if (const_node != nullptr)
    {
      nodefinder->enroll(i, const_node);
    }
  }
}

void OpSelector::import_operators(luci::GraphBuilderContext &gb_context)
{
  const luci::GraphBuilderSource *source_ptr = &luci::GraphBuilderRegistry::get();
  const auto &operators = _reader.operators();

  // Import the operators.
  // Note that operators in model are stored in execution order. This means that when importing
  // an operator, its input operators have already been imported. We exploit this fact to set up
  // node's inputs right after creating the node.
  for (uint32_t i = 0; i < operators.size(); ++i)
  {
    const circle::OperatorT &op = *operators[i];
    circle::BuiltinOperator builtincode = _reader.builtin_code(op);

    if (const auto *builder = source_ptr->lookup(builtincode))
    {
      luci::GraphBuilder::ValidateArgs args(op, _reader);
      if (!builder->validate(args))
      {
        throw oops::UserExn("Invalid operator", _reader.opcode_name(op));
      }

      auto built_op = builder->build(op, &gb_context);
      luci::set_node_id(built_op, i);

      add_origin(built_op, luci::single_origin(i, built_op->name()));
    }
    else
    {
      throw oops::UserExn("Not supported", _reader.opcode_name(op));
    }
  }
}

void OpSelector::create_graph_outputs(luci::GraphBuilderContext &gb_context, uint32_t output)
{
  auto graph = gb_context.graph();
  auto nodefinder = gb_context.nodefinder();

  const auto &tensors = _reader.tensors();
  auto tensors_ptr = _reader.tensors_ptr();
  assert(tensors_ptr != nullptr);

  // graph outputs
  const circle::TensorT &tensor = *tensors[output];

  auto output_node = graph->nodes()->create<luci::CircleOutput>();
  assert(output_node != nullptr);
  auto output_from = nodefinder->node(output);
  if (output_from != nullptr)
  {
    output_node->from(output_from);
  }
  else
  {
    // NOTE loco::Graph requires all input node(s) to a node should exist.
    //      Here, CircleOutput needs an input node.
    //      We add a dummy node to make it happy.
    auto output_dummy = graph->nodes()->create<luci::CircleOutputDummy>();
    assert(output_dummy != nullptr);
    output_node->from(output_dummy);

    luci::copy_tensor_attributes(tensor, output_dummy);
    if (tensors_ptr->Get(output)->shape() == nullptr)
    {
      output_dummy->shape_status(luci::ShapeStatus::NOSHAPE);
    }
    else
    {
      output_dummy->shape_status(luci::ShapeStatus::VALID);
    }
  }

  // set the graph output name and node object
  auto graph_output = graph->outputs()->create();
  std::string tname = luci::tensor_name(tensor);
  assert(tname.length() > 0);
  graph_output->name(tname);

  luci::copy_tensor_attributes(tensor, output_node);

  // Set GraphInputOutputIndex for graph
  output_node->index(graph_output->index());

  assert(tensor.shape_signature.size() == 0 ||
         tensor.shape_signature.size() == tensor.shape.size());

  // Shape of Output
  auto output_shape = std::make_unique<loco::TensorShape>();
  const std::vector<int32_t> &output_dims = tensor.shape; // in NHWC
  output_shape->rank(output_dims.size());
  for (uint32_t r = 0; r < output_dims.size(); ++r)
  {
    if (tensor.shape_signature.size() > 0 && tensor.shape_signature.at(r) == -1)
    {
      output_shape->dim(r).unset();
    }
    else
    {
      output_shape->dim(r).set(output_dims[r]);
    }
  }
  graph_output->shape(std::move(output_shape));

  // Data type
  auto dtype = luci::luci_datatype(tensor.type);
  graph_output->dtype(dtype);
}

std::unique_ptr<luci::Module>
OpSelector::select_nodes(std::vector<const luci::CircleNode *> selected_nodes)
{
  auto module = luci::make_module();

  const luci::GraphBuilderSource *source_ptr = &luci::GraphBuilderRegistry::get();

  for (uint32_t g = 0; g < _reader.num_subgraph(); ++g)
  {
    std::unique_ptr<loco::Graph> graph = loco::make_graph(); // create new empty graph

    assert(_reader.select_subgraph(g)); // select subgraph. usually, 0 is main
    graph->name(_reader.name());        // set name of graph (if it has one graph, name is empty.)

    auto nodefinder = std::make_unique<luci::IndexNodeFinder>();
    auto tensoroutputs = std::make_unique<luci::IndexTensorOutputs>();

    luci::GraphBuilderContext gb_context(graph.get(), &_reader, nodefinder.get(),
                                         tensoroutputs.get()); // graph build helper.
    if (g == MAIN_SUBGRAPH) // g is main subgraph, copy selected input, output nodes.
    {
      std::set<uint32_t> used_output_tensors;
      std::set<uint32_t> graph_inputs;
      std::set<uint32_t> graph_outputs;

      check_connected(selected_nodes, used_output_tensors, graph_inputs, graph_outputs);

      build_cache_outputs(gb_context);

      for (auto input : graph_inputs)
      {
        if (used_output_tensors.find(input) !=
            used_output_tensors.end()) // if it is virtual node, never used before.
        {
          create_graph_inputs(gb_context, input);
        }
      }

      create_circle_const(gb_context);

      import_operators(gb_context);

      for (auto output : graph_outputs)
      {
        create_graph_outputs(gb_context, output);
      }

      module->add(std::move(graph)); // add graph in module
    }
    else if (_has_subgraph) // g is not main, and main graph has while or if node, copy all input,
                            // output nodes.
    {
      build_cache_outputs(gb_context);

      for (const auto input : _reader.inputs())
      {
        create_graph_inputs(gb_context, input);
      }

      create_circle_const(gb_context);

      import_operators(gb_context);

      for (auto output : _reader.outputs())
      {
        create_graph_outputs(gb_context, output);
      }

      module->add(std::move(graph)); // add graph in module
    }
  }

  luci::post_import_graph(module.get(), _reader); // No this function, can't select if and while

  return module;
}

} // namespace opselector
