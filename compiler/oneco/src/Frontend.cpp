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

#include <moco/onnx/Frontend.h>

#include "Convert.h"
#include "GraphBuilderContext.h"
#include "GraphBuilderRegistry.h"
#include "Onnxutil.h"

#include <cwrap/Fildes.h>

#include <onnx/onnx.pb.h>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <sstream>
#include <string>
#include <stdexcept>

#include <fcntl.h>
#include <unistd.h>

namespace
{

bool load_text(const cwrap::Fildes &fildes, onnx::ModelProto &model_proto)
{
  google::protobuf::io::FileInputStream fis(fildes.get());

  return google::protobuf::TextFormat::Parse(&fis, &model_proto);
}

bool load_binary(const cwrap::Fildes &fildes, onnx::ModelProto &model_proto)
{
  google::protobuf::io::FileInputStream fis(fildes.get());
  google::protobuf::io::CodedInputStream cis(&fis);

  return model_proto.ParseFromCodedStream(&cis);
}

void load_onnx(const std::string &path, moco::onnx::Frontend::FileType type,
               onnx::ModelProto &model_proto)
{
  cwrap::Fildes fildes{open(path.c_str(), O_RDONLY)};

  if (fildes.get() < 0)
  {
    throw std::runtime_error{"Error: " + path + " not found"};
  }

  bool result = (type == moco::onnx::Frontend::FileType::Text) ? load_text(fildes, model_proto)
                                                               : load_binary(fildes, model_proto);

  if (!result)
  {
    throw std::runtime_error{"Error: Failed to parse " + path};
  }
}

// TODO Make comments clear
void convert_graph(::onnx::ModelProto &onnx_model_proto, loco::Graph *graph)
{
  auto nodes = std::make_unique<moco::onnx::SymbolTable>();
  auto input_names = std::make_unique<moco::onnx::SymbolTable>();

  moco::onnx::GraphBuilderContext gb_context(graph, nodes.get(), input_names.get());

  // Building a loco graph
  // 1. Convert onnx::node to loco::Node
  // 2. Convert onnx::initializer to loco::ConstGen node
  // 3. Convert onnx::input to loco::Pull node
  // 4. Connect inputs: set all node input(from a string) to actual node object
  // 5. Set graph input
  // 6. Create loco::Push node (with a proper input), and mark it as a graph output

  assert(onnx_model_proto.has_graph());
  ::onnx::GraphProto onnx_graph_proto = onnx_model_proto.graph();

  /// All nodes in the ModelProto's graph will bind against the operator
  /// with the same-domain/same-op_type operator with the HIGHEST version
  /// in the referenced operator sets.
  assert(onnx_model_proto.opset_import_size() > 0);
  int64_t opset_version = 1;
  for (int i = 0; i < onnx_model_proto.opset_import_size(); ++i)
  {
    auto opset = onnx_model_proto.opset_import(i);

    if (!opset.has_domain() || moco::onnx::is_default_domain(opset.domain()))
    {
      if (opset.version() > opset_version)
      {
        opset_version = opset.version();
      }
    }
    else
    {
      throw std::runtime_error("Not supported for custom operation");
    }
  }

  // 1. Convert all the nodes to loco::Node
  for (const auto &n : onnx_graph_proto.node())
  {
    if (const auto *graph_builder = moco::onnx::GraphBuilderRegistry::get().lookup(n.op_type()))
    {
      if (!graph_builder->validate(opset_version, n))
      {
        throw std::runtime_error{"Invalid operator: " + n.op_type()};
      }

      graph_builder->build(opset_version, n, &gb_context);
    }
    else
    {
      throw std::runtime_error{"Not supported: " + n.op_type()};
    }
  }

  // 2. Convert onnx::initializer to loco::ConstGen node
  std::set<std::string> initializer_name_set;
  for (int i = 0; i < onnx_graph_proto.initializer_size(); ++i)
  {
    auto initializer = onnx_graph_proto.initializer(i);

    initializer_name_set.insert(initializer.name());

    // TODO Support other data types
    auto data = moco::onnx::get_float_data(initializer);

    auto const_node = graph->nodes()->create<loco::ConstGen>();
    const_node->dtype(moco::onnx::as_loco_datatype(initializer.data_type()));
    const_node->rank(initializer.dims_size());
    // TODO Support other data types
    const_node->size<loco::DataType::FLOAT32>(data.size());

    for (uint32_t i = 0; i < const_node->rank(); ++i)
    {
      const_node->dim(i) = initializer.dims(i);
    }

    for (uint32_t i = 0; i < data.size(); ++i)
    {
      // TODO Support other data types
      const_node->at<loco::DataType::FLOAT32>(i) = data.at(i);
    }

    nodes->enroll(initializer.name(), const_node);
  }

  // 3. Convert onnx::input to loco::Pull node
  for (int i = 0; i < onnx_graph_proto.input_size(); i++)
  {
    auto input = onnx_graph_proto.input(i);

    // Already substituted by ConstGen node
    if (initializer_name_set.find(input.name()) != initializer_name_set.end())
      continue;

    auto pull_node = graph->nodes()->create<loco::Pull>();

    auto tensor = input.type().tensor_type();
    pull_node->dtype(moco::onnx::as_loco_datatype(tensor.elem_type()));
    pull_node->rank(tensor.shape().dim_size());
    for (uint32_t i = 0; i < pull_node->rank(); ++i)
    {
      pull_node->dim(i) = (uint32_t)tensor.shape().dim(i).dim_value();
    }

    nodes->enroll(input.name(), pull_node);
  }

  // 4. Connect inputs: set all node input(from a string) to actual node object
  loco::Graph::NodeContext *graph_nodes = graph->nodes();
  uint32_t nodes_count = graph_nodes->size();
  for (uint32_t n = 0; n < nodes_count; ++n)
  {
    loco::Node *node_to_set = graph_nodes->at(n);

    unsigned int names_size = input_names->size(node_to_set);
    assert(names_size == node_to_set->arity());
    for (unsigned int i = 0; i < names_size; ++i)
    {
      auto input_name = input_names->name(node_to_set, i);
      auto node = nodes->node(input_name);

      // TODO use enum instead of dynamic_cast
      loco::Forward *forward_node = dynamic_cast<loco::Forward *>(node_to_set);
      if (forward_node != nullptr)
        forward_node->input(node);
    }
  }

  // 5. Set graph input
  for (int i = 0; i < onnx_graph_proto.input_size(); i++)
  {
    auto input = onnx_graph_proto.input(i).name();

    // Already substituted by ConstGen node
    if (initializer_name_set.find(input) != initializer_name_set.end())
      continue;

    auto node = nodes->node(input);
    assert(node != nullptr);

    auto graph_input = graph->inputs()->create();

    loco::Pull *pull_node = dynamic_cast<loco::Pull *>(node);
    assert(pull_node != nullptr);

    graph_input->name(input);
    loco::link(graph_input, pull_node);
  }

  // 6. Create loco::Push node (with a proper input), and mark it as a graph output
  for (int i = 0; i < onnx_graph_proto.output_size(); i++)
  {
    auto output = onnx_graph_proto.output(i).name();

    auto output_node = nodes->node(output);
    assert(output_node);

    // create loco::Push for output of graph
    auto push_node = graph->nodes()->create<loco::Push>();
    push_node->from(output_node); // set input of Push to output node

    // set the graph output name and node object
    auto graph_output = graph->outputs()->create();
    graph_output->name(output);
    loco::link(graph_output, push_node);
  }
}

} // namespace

namespace moco
{
namespace onnx
{

Frontend::Frontend()
{
  // DO NOTHING
}

std::unique_ptr<loco::Graph> Frontend::load(const char *modelfile, FileType type) const
{
  ::onnx::ModelProto onnx_model_proto;

  load_onnx(modelfile, type, onnx_model_proto);

  auto graph = loco::make_graph();

  convert_graph(onnx_model_proto, graph.get());

  return std::move(graph);
}

} // namespace onnx
} // namespace moco
