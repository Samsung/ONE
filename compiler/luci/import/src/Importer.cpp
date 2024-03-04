/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Importer.h"
#include "CircleImportMetadata.h"
#include "PostImport.h"

#include "luci/Import/GraphBuilder.h"
#include "luci/Import/GraphBuilderContext.h"
#include "luci/Import/GraphBuilderRegistry.h"
#include "luci/Import/CircleReader.h"
#include "luci/Import/Nodes/CircleConst.h"
#include "luci/Import/Nodes/CircleVariable.h"

#include <luci/IR/Module.h>
#include <luci/IR/CircleNodes.h>
#include <luci/Profile/CircleNodeID.h>
#include <luci/Profile/CircleNodeOrigin.h>
#include <luci/Plan/CircleNodeExecutionPlan.h>
#include <luci/Log.h>
#include <luci/LogHelper.h>

#include <oops/InternalExn.h>
#include <oops/UserExn.h>

#include <memory>

namespace
{

void convert_graph(const luci::GraphBuilderSource &source, luci::CircleReader &reader,
                   loco::Graph *graph)
{
  LOGGER(l);

  auto nodefinder = std::make_unique<luci::IndexNodeFinder>();
  auto tensoroutputs = std::make_unique<luci::IndexTensorOutputs>();

  luci::GraphBuilderContext gb_context(graph, &reader, nodefinder.get(), tensoroutputs.get());

  const auto operators = reader.operators();
  const auto tensors = reader.tensors();
  assert(!tensors.null());
  auto circle_metadata = std::make_unique<luci::CircleImportMetadata>(reader);

  // build a cache to identify if a tensor is output of an operator
  // if this is set, we should not create a CircleConst for this tensor
  for (uint32_t i = 0; i < operators.size(); ++i)
  {
    const auto op = operators[i];
    assert(op != nullptr);
    const auto outputs = luci::wrap(op->outputs());

    for (uint32_t j = 0; j < outputs.size(); ++j)
    {
      auto tidx = outputs[j];
      tensoroutputs->enroll(tidx);
    }
  }

  // graph inputs; there are no input nodes in TFlite but just Tensors
  // creating virtual input nodes will make possible to connect nodes that uses them
  // all attributes of tensor should be copied to CircleInput node
  for (const auto input : reader.inputs())
  {
    auto input_node = graph->nodes()->create<luci::CircleInput>();
    assert(input_node != nullptr);
    const auto tensor = tensors[input];
    assert(tensor != nullptr);

    luci::copy_tensor_attributes(tensor, input_node);
    if (tensor->shape() == nullptr)
      input_node->shape_status(luci::ShapeStatus::NOSHAPE);
    else
      input_node->shape_status(luci::ShapeStatus::VALID);

    INFO(l) << "[luci] NodeFinder INPUT(" << input << ") = " << input_node << std::endl;
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

    const auto tensor_shape_signature = luci::wrap(tensor->shape_signature());
    const auto tensor_shape = luci::wrap(tensor->shape());
    assert(tensor_shape_signature.size() == 0 ||
           tensor_shape_signature.size() == tensor_shape.size());

    // Shape of GraphInput
    auto input_shape = std::make_unique<loco::TensorShape>();
    const auto &input_dims = tensor_shape; // in NHWC
    input_shape->rank(input_dims.size());
    for (uint32_t r = 0; r < input_dims.size(); ++r)
    {
      if (tensor_shape_signature.size() > 0 && tensor_shape_signature.at(r) == -1)
        input_shape->dim(r).unset();
      else
        input_shape->dim(r).set(input_dims[r]);
    }
    graph_input->shape(std::move(input_shape));
  }

  // Create CircleNodes for constant tensors.
  // NOTE Origin is intentionally not provided for constants.
  auto const_builder = source.lookup(luci::NodeBuilderType::BUFFER);
  if (not const_builder)
    throw oops::UserExn("Not supported", "tensor with buffer builder");

  for (uint32_t i = 0; i < tensors.size(); ++i)
  {
    auto *const_node = const_builder->build(i, &gb_context);
    if (const_node != nullptr)
      nodefinder->enroll(i, const_node);
  }

  // Create CircleVariable nodes for variable tensors
  // TODO Add Origin if needed, skip for now
  for (uint32_t i = 0; i < tensors.size(); ++i)
  {
    luci::CircleVariable *variable_node = luci::create_circlevariable(&gb_context, i);
    if (variable_node != nullptr)
      nodefinder->enroll(i, variable_node);
  }

  // Import the operators.
  // Note that operators in model are stored in execution order. This means that when importing
  // an operator, its input operators have already been imported. We exploit this fact to set up
  // node's inputs right after creating the node.
  auto origin_table = circle_metadata->origin_table();
  for (uint32_t i = 0; i < operators.size(); ++i)
  {
    const auto op = operators[i];
    assert(op != nullptr);
    circle::BuiltinOperator builtincode = reader.builtin_code(op);

    if (const auto *builder = source.lookup(builtincode))
    {
      // create temporary unpack API obj
      circle::OperatorT oper_t;
      op->UnPackTo(&oper_t);

      luci::GraphBuilder::ValidateArgs args(oper_t, reader);
      if (!builder->validate(args))
      {
        throw oops::UserExn("Invalid operator", reader.opcode_name(op));
      }

      auto built_op = builder->build(oper_t, &gb_context);
      set_node_id(built_op, i);
      if (origin_table.find(i) != origin_table.end())
        add_origin(built_op, origin_table.at(i));
      else
        add_origin(built_op, luci::single_origin(i, built_op->name()));
    }
    else
    {
      throw oops::UserExn("Not supported", reader.opcode_name(op));
    }
  }

  // graph outputs
  for (auto output : reader.outputs())
  {
    const auto tensor = tensors[output];
    assert(tensor != nullptr);

    auto output_node = graph->nodes()->create<luci::CircleOutput>();
    assert(output_node != nullptr);
    auto output_from = nodefinder->node(output);
    if (output_from != nullptr)
      output_node->from(output_from);
    else
    {
      // NOTE loco::Graph requires all input node(s) to a node should exist.
      //      Here, CircleOutput needs an input node.
      //      We add a dummy node to make it happy.
      auto output_dummy = graph->nodes()->create<luci::CircleOutputDummy>();
      assert(output_dummy != nullptr);
      output_node->from(output_dummy);

      luci::copy_tensor_attributes(tensor, output_dummy);
      if (tensor->shape() == nullptr)
        output_dummy->shape_status(luci::ShapeStatus::NOSHAPE);
      else
        output_dummy->shape_status(luci::ShapeStatus::VALID);
    }

    INFO(l) << "[luci] NodeFinder OUTPUT(" << output << ") = " << output_node << std::endl;

    // set the graph output name and node object
    auto graph_output = graph->outputs()->create();
    std::string tname = luci::tensor_name(tensor);
    assert(tname.length() > 0);
    graph_output->name(tname);

    luci::copy_tensor_attributes(tensor, output_node);

    // Set GraphInputOutputIndex for graph
    output_node->index(graph_output->index());

    const auto tensor_shape_signature = luci::wrap(tensor->shape_signature());
    const auto tensor_shape = luci::wrap(tensor->shape());
    assert(tensor_shape_signature.size() == 0 ||
           tensor_shape_signature.size() == tensor_shape.size());

    // Shape of Output
    auto output_shape = std::make_unique<loco::TensorShape>();
    const auto &output_dims = tensor_shape; // in NHWC
    output_shape->rank(output_dims.size());
    for (uint32_t r = 0; r < output_dims.size(); ++r)
    {
      if (tensor_shape_signature.size() > 0 && tensor_shape_signature.at(r) == -1)
        output_shape->dim(r).unset();
      else
        output_shape->dim(r).set(output_dims[r]);
    }
    graph_output->shape(std::move(output_shape));

    // Data type
    auto dtype = luci::luci_datatype(tensor->type());
    graph_output->dtype(dtype);
  }
}

class ValidateCollector final : public loco::ErrorListener
{
public:
  void notify(const loco::ErrorDetail<loco::ErrorCategory::MissingArgument> &d) override
  {
    LOGGER(l);
    INFO(l) << "[luci] GraphValidate error " << d.node() << "(" << d.index() << ")" << std::endl;
  }
};

} // namespace

namespace luci
{

Importer::Importer()
{
  // DO NOTHING
}

std::unique_ptr<loco::Graph> Importer::import(const circle::Model *model) const
{
  auto graph = loco::make_graph();

  const GraphBuilderSource *source_ptr = &GraphBuilderRegistry::get();

  if (_source != nullptr)
  {
    // Use user-defined GraphBuilderSource
    source_ptr = _source;
  }

  CircleReader reader;
  if (!reader.parse(model))
    return nullptr;

  if (reader.num_subgraph() != 1)
  {
    INTERNAL_EXN("Use 'importModule()' for multiple subgraphs");
  }
  if (!reader.select_subgraph(0))
    return nullptr;

  // Convert circle::Model to loco::Graph
  convert_graph(*source_ptr, reader, graph.get());

  LOGGER(l);
  VERBOSE(l, 3) << "--- graph dump begin -------------------------------------------";
  VERBOSE(l, 3) << "Name: " << graph->name();
  VERBOSE(l, 3) << fmt(graph.get());
  VERBOSE(l, 3) << "--- graph dump end ---------------------------------------------";

  assert(loco::valid(graph.get(), std::make_unique<ValidateCollector>()));

  return graph;
}

std::unique_ptr<Module> Importer::importModule(const circle::Model *model) const
{
  auto module = make_module();

  const GraphBuilderSource *source_ptr = &GraphBuilderRegistry::get();

  if (_source != nullptr)
  {
    // Use user-defined GraphBuilderSource
    source_ptr = _source;
  }

  CircleReader reader;
  if (!reader.parse(model))
    return nullptr;

  for (uint32_t g = 0; g < reader.num_subgraph(); ++g)
  {
    auto graph = loco::make_graph();

    if (!reader.select_subgraph(g))
      return nullptr;

    graph->name(reader.name());

    // Convert circle::Model to loco::Graph
    convert_graph(*source_ptr, reader, graph.get());

    LOGGER(l);
    VERBOSE(l, 3) << "--- graph dump begin -------------------------------------------";
    VERBOSE(l, 3) << "Name: " << graph->name();
    VERBOSE(l, 3) << fmt(graph.get());
    VERBOSE(l, 3) << "--- graph dump end ---------------------------------------------";

    assert(loco::valid(graph.get(), std::make_unique<ValidateCollector>()));

    module->add(std::move(graph));
  }

  post_import_graph(module.get(), reader);

  // Initialize 'source_table'
  auto circle_metadata = std::make_unique<luci::CircleImportMetadata>(reader);

  if (circle_metadata->map_tensors_indexes().size() > 0)
  {
    // If there is 'source_table' metadata in circle model, copy the table.
    module->map_tenros_indexes(circle_metadata->map_tensors_indexes());
  }

  if (circle_metadata->source_table().size() > 0)
  {
    // If there is 'source_table' metadata in circle model, copy the table.
    module->source_table(circle_metadata->source_table());
  }
  else
  {
    // If there is no 'source_table' metadata in circle model,
    // create new table with circle nodes.
    std::map<uint32_t, std::string> table;

    // NOTE Only first subgraph is considered
    for (auto node : loco::all_nodes(module->graph(0)))
    {
      auto circle_node = loco::must_cast<luci::CircleNode *>(node);

      // Virtual nodes may not have id
      if (!has_node_id(circle_node))
        continue;

      assert(table.find(get_node_id(circle_node)) == table.end());
      table.insert({get_node_id(circle_node), circle_node->name()});
    }

    module->source_table(table);
  }

  // Add execution_plan annotations
  if (circle_metadata->execution_plan_table().size() > 0)
  {
    auto execution_plan_table = circle_metadata->execution_plan_table();
    auto node_position = 0;
    for (auto node : loco::postorder_traversal(loco::output_nodes(module->graph())))
    {
      if (auto circle_node = dynamic_cast<luci::CircleNode *>(node))
      {
        if (execution_plan_table.count(node_position) == 0)
          continue;

        auto node_plan = execution_plan_table[node_position];
        assert(node_plan.size() > 0);

        luci::add_execution_plan(
          circle_node,
          luci::CircleNodeExecutionPlan(
            node_plan[0], std::vector<uint32_t>(node_plan.begin() + 1, node_plan.end())));
      }
      node_position++;
    }
  }

  return module;
}

} // namespace luci
