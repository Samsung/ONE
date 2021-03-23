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
#include "PostImport.h"

#include "luci/Import/GraphBuilder.h"
#include "luci/Import/GraphBuilderContext.h"
#include "luci/Import/GraphBuilderRegistry.h"
#include "luci/Import/CircleReader.h"
#include "luci/Import/Nodes/CircleConst.h"

#include <luci/IR/Module.h>
#include <luci/IR/CircleNodes.h>
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

    luci::GraphBuilderContext gb_context(graph, &reader, nodefinder.get());

    const auto &operators = reader.operators();
    const auto &tensors = reader.tensors();

    // graph inputs; there are no input nodes in TFlite but just Tensors
    // creating virtual input nodes will make possible to connect nodes that uses them
    // all attributes of tensor should be copied to CircleInput node
    for (const auto input : reader.inputs())
    {
      auto input_node = graph->nodes()->create<luci::CircleInput>();
      assert(input_node != nullptr);
      const circle::TensorT &tensor = *tensors[input];

      luci::copy_tensor_attributes(tensor, input_node);

      INFO(l) << "[luci] NodeFinder INPUT(" << input << ") = " << input_node << std::endl;
      nodefinder->enroll(input, input_node);

      // Name
      auto graph_input = graph->inputs()->create();
      graph_input->name(input_node->name());

      // Set GraphInputOutputIndex for graph
      input_node->index(graph_input->index());

      // Data type
      graph_input->dtype(input_node->dtype());

      // Shape of GraphInput
      auto input_shape = std::make_unique<loco::TensorShape>();
      const std::vector<int32_t> &input_dims = tensor.shape; // in NHWC
      input_shape->rank(input_dims.size());
      for (uint32_t r = 0; r < input_dims.size(); ++r)
        input_shape->dim(r) = loco::Dimension(input_dims[r]);
      graph_input->shape(std::move(input_shape));
    }

    // Create CircleConst nodes for constant tensors.
    const auto &buffers = reader.buffers();
    for (uint32_t i = 0; i < tensors.size(); ++i)
    {
      const circle::TensorT &tensor = *tensors[i];
      const std::vector<uint8_t> &buffer = buffers[tensor.buffer]->data;
      if (!buffer.empty())
      {
        luci::CircleConst *const_node = luci::create_circleconst(&gb_context, i);
        nodefinder->enroll(i, const_node);
      }
    }

    // Import the operators.
    // Note that operators in model are stored in execution order. This means that when importing
    // an operator, its input operators have already been imported. We exploit this fact to set up
    // node's inputs right after creating the node.
    for (uint32_t i = 0; i < operators.size(); ++i)
    {
      const circle::OperatorT &op = *operators[i];
      circle::BuiltinOperator builtincode = reader.builtin_code(op);

      if (const auto *builder = source.lookup(builtincode))
      {
        luci::GraphBuilder::ValidateArgs args(op, reader);
        if (!builder->validate(args))
        {
          throw oops::UserExn("Invalid operator", reader.opcode_name(op));
        }

        builder->build(op, &gb_context);
      }
      else
      {
        throw oops::UserExn("Not supported", reader.opcode_name(op));
      }
    }

    // graph outputs
    for (auto output : reader.outputs())
    {
      const circle::TensorT &tensor = *tensors[output];

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
      }

      INFO(l) << "[luci] NodeFinder OUTPUT(" << output << ") = " << output_node << std::endl;

      // set the graph output name and node object
      auto graph_output = graph->outputs()->create();
      std::string tname = luci::tensor_name(tensor);
      graph_output->name("output_" + tname);

      luci::copy_tensor_attributes(tensor, output_node);

      // Set GraphInputOutputIndex for graph
      output_node->index(graph_output->index());

      // Shape of Output
      auto output_shape = std::make_unique<loco::TensorShape>();
      const std::vector<int32_t> &output_dims = tensor.shape; // in NHWC
      output_shape->rank(output_dims.size());
      for (uint32_t r = 0; r < output_dims.size(); ++r)
        output_shape->dim(r) = loco::Dimension(output_dims[r]);
      graph_output->shape(std::move(output_shape));

      // Data type
      auto dtype = luci::luci_datatype(tensor.type);
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
    // INFO(l) << "--- graph dump begin ---" << std::endl;
    INFO(l) << "Name: " << std::string(graph->name()) << std::endl;
    // INFO(l) << fmt(graph.get());
    // INFO(l) << "--- graph dump end ---" << std::endl;

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
      // INFO(l) << "--- graph dump begin ---" << std::endl;
      INFO(l) << "Name: " << std::string(graph->name()) << std::endl;
      // INFO(l) << fmt(graph.get());
      // INFO(l) << "--- graph dump end ---" << std::endl;

      assert(loco::valid(graph.get(), std::make_unique<ValidateCollector>()));

      module->add(std::move(graph));
    }

    post_import_graph(module.get(), reader);

    return module;
  }

} // namespace luci
