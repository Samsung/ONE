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
#include "luci/../../src/CircleImportMetadata.h"
#include "luci/../../src/PostImport.h"

#include "luci/Service/ChangeOutputs.h"

#include "luci/Import/GraphBuilder.h"
#include "luci/Import/GraphBuilderContext.h"
#include "luci/Import/GraphBuilderRegistry.h"
#include "luci/Import/CircleReader.h"
#include "luci/Import/Nodes/CircleConst.h"

#include <loco/IR/Graph.h>
#include <luci/IR/CircleNode.h>

#include <oops/UserExn.h>

#include <cassert>
#include <iostream>
#include <map>

namespace opselector
{

void convert_graph(const luci::GraphBuilderSource &source, luci::CircleReader &reader,
                   loco::Graph *graph)
{

  auto nodefinder = std::make_unique<luci::IndexNodeFinder>();
  auto tensoroutputs = std::make_unique<luci::IndexTensorOutputs>();

  luci::GraphBuilderContext gb_context(graph, &reader, nodefinder.get(), tensoroutputs.get());

  const auto &operators = reader.operators();
  const auto &tensors = reader.tensors();
  auto tensors_ptr = reader.tensors_ptr();
  assert(tensors_ptr != nullptr);
  auto circle_metadata = std::make_unique<luci::CircleImportMetadata>(reader);

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

  // graph inputs; there are no input nodes in TFlite but just Tensors
  // creating virtual input nodes will make possible to connect nodes that uses them
  // all attributes of tensor should be copied to CircleInput node
  for (const auto input : reader.inputs())
  {
    auto input_node = graph->nodes()->create<luci::CircleInput>();
    assert(input_node != nullptr);
    const circle::TensorT &tensor = *tensors[input];

    luci::copy_tensor_attributes(tensor, input_node);
    if (tensors_ptr->Get(input)->shape() == nullptr)
      input_node->shape_status(luci::ShapeStatus::NOSHAPE);
    else
      input_node->shape_status(luci::ShapeStatus::VALID);

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
        input_shape->dim(r).unset();
      else
        input_shape->dim(r).set(input_dims[r]);
    }
    graph_input->shape(std::move(input_shape));
  }

  // Create CircleConst nodes for constant tensors.
  for (uint32_t i = 0; i < tensors.size(); ++i)
  {
    luci::CircleConst *const_node = luci::create_circleconst(&gb_context, i);
    if (const_node != nullptr)
      nodefinder->enroll(i, const_node);
  }

  // Import the operators.
  // Note that operators in model are stored in execution order. This means that when importing
  // an operator, its input operators have already been imported. We exploit this fact to set up
  // node's inputs right after creating the node.
  auto origin_table = circle_metadata->origin_table();
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

      auto built_op = builder->build(op, &gb_context);
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
      if (tensors_ptr->Get(output)->shape() == nullptr)
        output_dummy->shape_status(luci::ShapeStatus::NOSHAPE);
      else
        output_dummy->shape_status(luci::ShapeStatus::VALID);
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
        output_shape->dim(r).unset();
      else
        output_shape->dim(r).set(output_dims[r]);
    }
    graph_output->shape(std::move(output_shape));

    // Data type
    auto dtype = luci::luci_datatype(tensor.type);
    graph_output->dtype(dtype);
  }
}

std::unique_ptr<luci::Module> OpSelector::select_nodes(const circle::Model *circle_model,
                                           std::map<uint32_t, std::string> &id_name_selected_nodes)
{
  auto module = luci::make_module();

  const luci::GraphBuilderSource *source_ptr = &luci::GraphBuilderRegistry::get();
  luci::CircleReader reader;

  assert(reader.parse(circle_model));

  for (uint32_t g = 0; g < 1; ++g)
  {
    std::unique_ptr<loco::Graph> graph = loco::make_graph();
    std::vector<const circle::OperatorT*> selected_operators;

    reader.select_subgraph(g);
    
    graph->name(reader.name());

    std::cout<<"Subgraph Name: "<< reader.name() << std::endl;

    const auto &opers = reader.operators(); // 오퍼레이터는 우리가 netron을 통해 확인할 수 있는 그 노드들의 tensor()상의 인덱스를 저장하고 있음
    const auto &tense = reader.tensors();
    auto tensors_ptr = reader.tensors_ptr();
    assert(tensors_ptr != nullptr);
    auto circle_metadata = std::make_unique<luci::CircleImportMetadata>(reader);

    auto nodefinder = std::make_unique<luci::IndexNodeFinder>();
    auto tensoroutputs = std::make_unique<luci::IndexTensorOutputs>();

    luci::GraphBuilderContext gb_context(graph.get(), &reader, nodefinder.get(), tensoroutputs.get());

    for(auto iter = id_name_selected_nodes.begin(); iter != id_name_selected_nodes.end(); iter++)
    {
      selected_operators.push_back(opers[iter->first].get());

      std::cout << "============== Type: " << tense[opers[iter->first]->outputs[0]]->type << " ==============" << std::endl;
      std::cout << "    <INPUT>" << std::endl;
      for(auto node : opers[iter->first].get()->inputs)
        std::cout<<"operator["<< iter->first << "] id: " << node <<" " << "input: " << tense[node]->name << std::endl;
      std::cout << "    <OUTPUT>" << std::endl;
      for(auto node : opers[iter->first].get()->outputs)
        std::cout<<"operator["<< iter->first << "] id: " << node <<" " << "output: " << tense[node]->name << std::endl;
      std::cout<<std::endl;
    }

    // build a cache to identify if a tensor is output of an operator
    // if this is set, we should not create a CircleConst for this tensor
    for (uint32_t i = 0; i < opers.size(); ++i)
    {
      const circle::OperatorT &op = *opers[i];
      const auto &outputs = op.outputs;

      for (uint32_t j = 0; j < outputs.size(); ++j)
      {
        auto tidx = outputs[j];
        tensoroutputs->enroll(tidx);
      }
    }
    
    //for (const auto input : selected_operators[0]->inputs)
    for (int a=0; a<1; a++)
    {
      int input=selected_operators[0]->inputs[0];
      auto input_node = graph->nodes()->create<luci::CircleInput>();
      assert(input_node != nullptr);
      const circle::TensorT &tensor = *tense[input];

      luci::copy_tensor_attributes(tensor, input_node);
      if (tensors_ptr->Get(input)->shape() == nullptr)
        input_node->shape_status(luci::ShapeStatus::NOSHAPE);
      else
        input_node->shape_status(luci::ShapeStatus::VALID);

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
          input_shape->dim(r).unset();
        else
          input_shape->dim(r).set(input_dims[r]);
      }
      graph_input->shape(std::move(input_shape));
    }
    // Create CircleConst nodes for constant tensors.
    for (uint32_t i = 0; i < tense.size(); ++i)
    {
      luci::CircleConst *const_node = luci::create_circleconst(&gb_context, i);
      if (const_node != nullptr)
        nodefinder->enroll(i, const_node);
    }

    
    // Import the operators.
    // Note that operators in model are stored in execution order. This means that when importing
    // an operator, its input operators have already been imported. We exploit this fact to set up
    // node's inputs right after creating the node.
    auto origin_table = circle_metadata->origin_table();
    for (uint32_t i = 0; i < opers.size(); ++i)
    {
      const circle::OperatorT &op = *opers[i];
      circle::BuiltinOperator builtincode = reader.builtin_code(op);

      if (const auto *builder = source_ptr->lookup(builtincode))
      {
        luci::GraphBuilder::ValidateArgs args(op, reader);
        if (!builder->validate(args))
        {
          throw oops::UserExn("Invalid operator", reader.opcode_name(op));
        }

        auto built_op = builder->build(op, &gb_context);
        luci::set_node_id(built_op, i);
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
    for (auto output : selected_operators[selected_operators.size()-1]->outputs)
    {
      const circle::TensorT &tensor = *tense[output];

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
        if (tensors_ptr->Get(output)->shape() == nullptr)
          output_dummy->shape_status(luci::ShapeStatus::NOSHAPE);
        else
          output_dummy->shape_status(luci::ShapeStatus::VALID);
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
          output_shape->dim(r).unset();
        else
          output_shape->dim(r).set(output_dims[r]);
      }
      graph_output->shape(std::move(output_shape));

      // Data type
      auto dtype = luci::luci_datatype(tensor.type);
      graph_output->dtype(dtype);
    }

    module->add(std::move(graph));
  }
  for (uint32_t g = 1; g < reader.num_subgraph(); ++g)
  {
    auto graph = loco::make_graph();
    
    if (!reader.select_subgraph(g))
      return nullptr;

    graph->name(reader.name());

    // Convert circle::Model to loco::Graph
    convert_graph(*source_ptr, reader, graph.get());

    module->add(std::move(graph));
  }

  luci::post_import_graph(module.get(), reader);

  return module;
}

} // namespace opselector
