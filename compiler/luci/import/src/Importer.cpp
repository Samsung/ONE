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

#include "luci/Import/GraphBuilder.h"
#include "luci/Import/GraphBuilderContext.h"
#include "luci/Import/GraphBuilderRegistry.h"
#include "luci/Import/CircleReader.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Log.h>
#include <luci/LogHelper.h>

#include <oops/UserExn.h>
#include <stdex/Memory.h>

namespace
{

void convert_graph(const luci::GraphBuilderSource &source, luci::CircleReader &reader,
                   loco::Graph *graph)
{
  LOGGER(l);

  auto opfinder = stdex::make_unique<luci::NodeOpFinder>();
  auto tensorfinder = stdex::make_unique<luci::NodeTensorFinder>();
  auto nodefinder = stdex::make_unique<luci::IndexNodeFinder>();
  auto updates = stdex::make_unique<luci::UpdateQueue>();

  luci::GraphBuilderContext gb_context(graph, &reader, opfinder.get(), tensorfinder.get(),
                                       nodefinder.get(), updates.get());

  auto operators = reader.operators();
  auto tensors = reader.tensors();

  // graph inputs; there are no input nodes in TFlite but just Tensors
  // creating virtual input nodes will make possible to connect nodes that uses them
  // all attributes of tensor should be copied to CircleInput node
  for (const auto input : reader.inputs())
  {
    auto input_node = graph->nodes()->create<luci::CircleInput>();
    assert(input_node != nullptr);
    opfinder->enroll(input_node, nullptr); // there is no Op for graph output
    auto tensor = tensors->Get(input);
    tensorfinder->enroll(input_node, tensor);

    auto tname = luci::tensor_name(tensor);
    input_node->name(tname);
    auto quantization = luci::tensor_quantization(tensor);
    if (quantization)
    {
      auto quantparam = luci::luci_quantparam(quantization);
      if (quantparam.get())
        input_node->quantparam(std::move(quantparam));
    }

    INFO(l) << "[luci] NodeFinder INPUT(" << input << ") = " << input_node << std::endl;
    nodefinder->enroll(input, input_node);

    // Shape of Input
    assert(tensor->shape());
    std::vector<int32_t> input_dims = luci::as_index_vector(tensor->shape()); // in NHWC
    input_node->rank(input_dims.size());
    for (uint32_t r = 0; r < input_dims.size(); ++r)
      input_node->dim(r) = loco::Dimension(input_dims[r]);

    // Data type of Input
    auto dtype = luci::luci_datatype(tensor);
    input_node->dtype(dtype);

    // Name
    auto graph_input = graph->inputs()->create();
    graph_input->name(tname);

    // Set GraphInputOutputIndex for graph
    input_node->index(graph_input->index());

    // Data type
    graph_input->dtype(dtype);
  }

  for (uint32_t i = 0; i < operators->Length(); ++i)
  {
    const auto op = operators->Get(i);
    circle::BuiltinOperator builtincode = reader.builtin_code(op);

    if (const auto *builder = source.lookup(builtincode))
    {
      if (!builder->validate(op))
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

  // connect nodes
  for (auto &update : updates->queue())
  {
    update->update(&gb_context);
  }

  // graph outputs
  for (auto output : reader.outputs())
  {
    auto output_node = graph->nodes()->create<luci::CircleOutput>();
    assert(output_node != nullptr);
    auto node = nodefinder->node(output);
    assert(node != nullptr);
    output_node->from(node);

    INFO(l) << "[luci] NodeFinder OUTPUT(" << output << ") = " << output_node << std::endl;

    // set the graph output name and node object
    auto tensor = tensors->Get(output);
    auto graph_output = graph->outputs()->create();
    std::string tname = luci::tensor_name(tensor);
    graph_output->name("output_" + tname);

    // Set GraphInputOutputIndex for graph
    output_node->index(graph_output->index());

    // Shape of Output
    assert(tensor->shape());
    auto output_shape = stdex::make_unique<loco::TensorShape>();
    std::vector<int32_t> output_dims = luci::as_index_vector(tensor->shape()); // in NHWC
    output_shape->rank(output_dims.size());
    for (uint32_t r = 0; r < output_dims.size(); ++r)
      output_shape->dim(r) = loco::Dimension(output_dims[r]);
    graph_output->shape(std::move(output_shape));

    // Data type
    auto dtype = luci::luci_datatype(tensor);
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

  // TODO support multiple subgraph when Circle supports
  assert(reader.num_subgraph() == 1);
  if (!reader.select_subgraph(0))
    return nullptr;

  // Convert circle::Model to loco::Graph
  convert_graph(*source_ptr, reader, graph.get());

  LOGGER(l);
  INFO(l) << fmt(graph.get());

  assert(loco::valid(graph.get(), stdex::make_unique<ValidateCollector>()));

  return std::move(graph);
}

} // namespace luci
