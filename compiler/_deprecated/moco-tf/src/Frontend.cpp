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

#include <moco/tf/Frontend.h>
#include <moco/Importer.h>
#include <moco/IR/TFNode.h>
#include <moco/Log.h>

#include <moco/Import/GraphBuilderRegistry.h>

#include "Canonicalizer.h"
#include "Optimizer.h"
#include "TFOptimizer.h"

#include "Transforms.h"

#include "Op/COpCall.h"

#include <loco/Service/ShapeInference.h>

#include <oops/UserExn.h>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <memory>
#include <iostream>
#include <sstream>
#include <fstream>
#include <stdexcept>

#include <fcntl.h>
#include <unistd.h>

namespace
{

bool load_text(std::istream *stream, tensorflow::GraphDef &graph_def)
{
  google::protobuf::io::IstreamInputStream iis(stream);

  return google::protobuf::TextFormat::Parse(&iis, &graph_def);
}

bool load_binary(std::istream *stream, tensorflow::GraphDef &graph_def)
{
  google::protobuf::io::IstreamInputStream iis(stream);
  google::protobuf::io::CodedInputStream cis(&iis);

  return graph_def.ParseFromCodedStream(&cis);
}

void load_tf(std::istream *stream, moco::tf::Frontend::FileType type,
             tensorflow::GraphDef &graph_def)
{
  bool result = (type == moco::tf::Frontend::FileType::Text) ? load_text(stream, graph_def)
                                                             : load_binary(stream, graph_def);
  if (!result)
  {
    throw oops::UserExn("Failed to parse prototxt from stream");
  }
}

// If Placeholder has no shape attribute, set unknown_rank property to true.
void set_unknown_rank(tensorflow::GraphDef &tf_graph_def)
{
  for (auto &n : *tf_graph_def.mutable_node())
  {
    if (n.op().compare("Placeholder"))
      continue;

    auto iter = n.attr().find("shape");
    if (iter == n.attr().end())
    {
      tensorflow::AttrValue attr;
      attr.mutable_shape()->set_unknown_rank(true);
      n.mutable_attr()->insert({"shape", attr});
    }
  }
}

/**
 * @brief Set input shape according to signature if node has unknown shape in GraphDef.
 *
 * @note If shape you provided is wrong or not enough, it returns false.
 */
bool set_input_shape(const moco::ModelSignature &signature, tensorflow::GraphDef &tf_graph_def)
{
  for (auto &n : *tf_graph_def.mutable_node())
  {
    if (n.op().compare("Placeholder"))
      continue;

    auto node_shape = n.mutable_attr()->at("shape").mutable_shape();
    auto sig_shape = signature.shape(n.name() + ":0");

    if (node_shape->unknown_rank() || !node_shape->dim_size())
    {
      // If shape in GraphDef is unknown, user must provide the shape info.
      if (sig_shape == nullptr)
        return false;
      node_shape->clear_unknown_rank();
      for (uint32_t i = 0; i < sig_shape->rank(); i++)
        node_shape->add_dim()->set_size(-1);
    }

    for (uint32_t d = 0; d < node_shape->dim_size(); d++)
    {
      if (node_shape->mutable_dim(d)->size() == -1)
      {
        if (sig_shape == nullptr)
          return false;
        node_shape->mutable_dim(d)->set_size(sig_shape->dim(d));
      }
      else
      {
        // If User provide shape info though it already exists in GraphDef, make sure it matches
        // the shape of GraphDef.
        if (sig_shape && node_shape->dim(d).size() != sig_shape->dim(d))
          return false;
      }
    }
  }
  return true;
}

void transform_tf(const moco::ModelSignature &signature, tensorflow::GraphDef &tf_graph_def)
{
  set_unknown_rank(tf_graph_def);
  if (!set_input_shape(signature, tf_graph_def))
    oops::UserExn("Info you provided may be wrong or not enough. Please check the info file.");
}

/**
 * @brief Returns GraphBuilderRegistry that looks up default registry and additions
 *        such as custom op
 */
moco::GraphBuilderRegistry make_graph_builder_registry(const moco::ModelSignature &sig)
{
  moco::GraphBuilderRegistry registry{&moco::GraphBuilderRegistry::get()};

  // build a COpCallGraphBuilder per custom op type
  for (const auto &custom_op : sig.customops())
  {
    std::unique_ptr<moco::tf::COpCallGraphBuilder> builder =
      std::make_unique<moco::tf::COpCallGraphBuilder>(&sig);
    registry.add(custom_op, std::move(builder));
  }

  return registry;
}

} // namespace

// TODO Find a proper place for this function

namespace
{

loco::TensorShape tensor_shape(loco::Node *node)
{
  assert(loco::shape_known(node));
  auto node_shape = loco::shape_get(node);
  return node_shape.as<loco::TensorShape>();
}

} // namespace

namespace moco
{
namespace tf
{

Frontend::Frontend()
{
  // DO NOTHING
}

std::unique_ptr<loco::Graph> Frontend::load(const ModelSignature &signature, const char *modelfile,
                                            FileType type) const
{
  // Using c++ standard library, rather than file descriptor, makes these lines portable
  std::ifstream ifs{modelfile, std::ios::in | std::ios::binary};
  return load(signature, &ifs, type);
}

std::unique_ptr<loco::Graph> Frontend::load(const ModelSignature &signature, std::istream *stream,
                                            FileType type) const
{
  tensorflow::GraphDef tf_graph_def;

  load_tf(stream, type, tf_graph_def);

  transform_tf(signature, tf_graph_def);

  auto graph = import(signature, tf_graph_def);

  return std::move(graph);
}

std::unique_ptr<loco::Graph> Frontend::import(const ModelSignature &signature,
                                              tensorflow::GraphDef &tf_graph_def) const
{
  LOGGER(frontend);

  // Let's use GraphBuilderRegistry with COpCallGraphBuilder
  GraphBuilderRegistry registry = make_graph_builder_registry(signature);

  Importer importer{&registry};

  INFO(frontend) << ">>";
  INFO(frontend) << ">> Import stage started";
  INFO(frontend) << ">>";
  auto graph = importer.import(signature, tf_graph_def);

  TFOptimizer tfoptimizier;

  // Transform TFNodes
  INFO(frontend) << ">>";
  INFO(frontend) << ">> TF optimize stage started";
  INFO(frontend) << ">>";
  tfoptimizier.optimize(graph.get());

  // Fill graph-level input/output shape
  //
  // ASSUMPTION! All the shapes are known at this point
  for (uint32_t n = 0; n < graph->inputs()->size(); ++n)
  {
    auto input = graph->inputs()->at(n);
    auto input_node = moco::placeholder_node(graph.get(), n);
    assert(input_node != nullptr);
    input->shape(std::make_unique<loco::TensorShape>(tensor_shape(input_node)));
  }

  for (uint32_t n = 0; n < graph->outputs()->size(); ++n)
  {
    auto output = graph->outputs()->at(n);
    auto output_node = moco::push_node(graph.get(), n);
    assert(output_node != nullptr);
    output->shape(std::make_unique<loco::TensorShape>(::tensor_shape(output_node)));
  }

  // Convert graph to hold only Canonical dialect
  Canonicalizer canonicalizer;

  INFO(frontend) << ">>";
  INFO(frontend) << ">> Canonicalize stage started";
  INFO(frontend) << ">>";
  canonicalizer.canonicalize(graph.get());

  // Optimize imported loco::Graph
  Optimizer optimizer;

  INFO(frontend) << ">>";
  INFO(frontend) << ">> Canonical optimize stage started";
  INFO(frontend) << ">>";
  optimizer.optimize(graph.get());

  return std::move(graph);
}

} // namespace tf
} // namespace moco
