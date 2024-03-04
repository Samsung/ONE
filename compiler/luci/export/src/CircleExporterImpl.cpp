/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "CircleExporterImpl.h"
#include "Optimize.h"
#include "CircleExportMetadata.h"
#include "CircleTensorExporter.h"
#include "CircleOperationExporter.h"
#include "CircleExporterUtils.h"

#include <luci/IR/CircleNodes.h>

#include <oops/InternalExn.h>
#include <mio/circle/schema_generated.h>
#include <flatbuffers/flatbuffers.h>

#include <cassert>
#include <unordered_map>
#include <string>
#include <vector>

namespace
{

void registerGraphInputTensors(loco::Graph *graph, luci::SubGraphContext &ctx)
{
  for (uint32_t n = 0; n < graph->inputs()->size(); ++n)
  {
    auto node = luci::input_node(graph, n);
    assert(node != nullptr);
    ctx._inputs.push_back(luci::get_tensor_index(node));
  }
}

void registerGraphOutputTensors(loco::Graph *graph, luci::SubGraphContext &ctx)
{
  for (uint32_t n = 0; n < graph->outputs()->size(); ++n)
  {
    auto push = luci::output_node(graph, n);
    assert(push != nullptr);
    auto node = push->from();
    assert(node != nullptr);

    // Do not export CircleOutput when it's input is CircleOutputExclude
    if (dynamic_cast<luci::CircleOutputExclude *>(push->from()) != nullptr)
    {
      continue;
    }

    ctx._outputs.push_back(luci::get_tensor_index(node));
  }
}

} // namespace

namespace
{

using namespace circle;
using namespace flatbuffers;

Offset<Vector<Offset<OperatorCode>>>
encodeOperatorCodes(FlatBufferBuilder &builder, std::unordered_map<luci::OpCode, uint32_t> &opcodes)
{
  std::vector<Offset<OperatorCode>> operator_codes_vec(opcodes.size());
  for (auto it : opcodes)
  {
    uint32_t idx = it.second;
    int8_t dep_code = 127; // BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES
    if (it.first.opcode < BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES)
      dep_code = static_cast<int8_t>(it.first.opcode);
    if (it.first.opcode != BuiltinOperator_CUSTOM)
    {
      operator_codes_vec[idx] =
        CreateOperatorCode(builder, dep_code, 0, it.first.version, it.first.opcode);
    }
    else
    {
      operator_codes_vec[idx] =
        CreateOperatorCode(builder, dep_code, builder.CreateString(it.first.custom_code),
                           it.first.version, it.first.opcode);
    }
  }

  return builder.CreateVector(operator_codes_vec);
}

} // namespace

namespace luci
{

using namespace circle;
using namespace flatbuffers;

CircleExporterImpl::CircleExporterImpl(loco::Graph *graph) { exportGraph(graph); }
CircleExporterImpl::CircleExporterImpl(Module *module) { exportModule(module); }

::flatbuffers::Offset<::circle::SubGraph>
CircleExporterImpl::exportSubgraph(SerializedGraphData &gd)
{
  auto tensors = _builder.CreateVector(gd._tensors);
  auto inputs = _builder.CreateVector(gd._inputs);
  auto outputs = _builder.CreateVector(gd._outputs);
  auto operators = _builder.CreateVector(gd._operators);
  auto name = _builder.CreateString(gd._name);
  auto subgraph = CreateSubGraph(_builder, tensors, inputs, outputs, operators, name);
  return subgraph;
}

void CircleExporterImpl::exportGraph(loco::Graph *graph)
{
  // do graph optimization
  optimize(graph);

  _builder.Clear();

  SerializedModelData md;
  SerializedGraphData gd;

  // This version is taken from comment in fbs
  constexpr uint32_t version = 0;

  // set Subgraph name
  gd._name = graph->name();

  // TODO set this value properly
  gd._data_format = circle::DataFormat::DataFormat_CHANNELS_LAST;

  // prepare model data
  prepareModelData(_builder, md);

  // parse graph into SerializedModelData structure
  exportOpDefinedTensors(graph, _builder, md, gd);

  // NOTE Invoke these register functions only after each node is annotated with its tensor_index
  registerGraphInputTensors(graph, gd);
  registerGraphOutputTensors(graph, gd);

  exportNodes(graph, _builder, md, gd);

  // encode operator codes
  auto operator_codes = encodeOperatorCodes(_builder, md._operator_codes);

  // Subgraphs
  Offset<SubGraph> subgraph = exportSubgraph(gd);
  auto subgraphs = _builder.CreateVector(std::vector<Offset<SubGraph>>{subgraph});

  // Description
  std::string description_str = "nnpackage";
  auto description = _builder.CreateString(description_str);

  // Metadata
  auto metadata_vec = createCircleMetadataVector(_builder, md);
  auto metadata = _builder.CreateVector(std::vector<Offset<Metadata>>(metadata_vec));

  // create array of buffers
  auto buffers = _builder.CreateVector(md._buffers);

  // Model
  auto model_offset = CreateModel(_builder, version, operator_codes, subgraphs, description,
                                  buffers, 0 /* metadata_buffer */, metadata);
  FinishModelBuffer(_builder, model_offset);
}

void CircleExporterImpl::exportModule(Module *module)
{
  assert(module->size() > 0);
  // do graph optimization

  SerializedModelData md;

  _builder.Clear();

  // prepare model data
  prepareModelData(_builder, md);

  std::vector<flatbuffers::Offset<circle::SubGraph>> subgraph_vec;

  for (size_t g = 0; g < module->size(); ++g)
  {
    auto graph = module->graph(g);

    optimize(graph);

    SerializedGraphData gd;

    // set Subgraph name
    gd._name = graph->name();

    // parse graph into SerializedModelData structure
    exportOpDefinedTensors(graph, _builder, md, gd);

    // NOTE Invoke these register functions only after each node is annotated with its tensor_index
    registerGraphInputTensors(graph, gd);
    registerGraphOutputTensors(graph, gd);

    exportNodes(graph, _builder, md, gd);

    // Subgraphs
    Offset<SubGraph> subgraph = exportSubgraph(gd);
    subgraph_vec.push_back(subgraph);
  }

  auto subgraphs = _builder.CreateVector(std::vector<Offset<SubGraph>>{subgraph_vec});

  // encode operator codes
  auto operator_codes = encodeOperatorCodes(_builder, md._operator_codes);

  // Description
  std::string description_str = "nnpackage";
  auto description = _builder.CreateString(description_str);

  // Metadata
  md._metadata.source_table(module->source_table());
  if (!module->map_tenros_indexes().empty())
    md._metadata.map_tensors_indexes(module->map_tenros_indexes());
  auto metadata_vec = createCircleMetadataVector(_builder, md);
  auto metadata = _builder.CreateVector(std::vector<Offset<Metadata>>(metadata_vec));

  // create array of buffers
  auto buffers = _builder.CreateVector(md._buffers);

  // This version is taken from comment in fbs
  constexpr uint32_t version = 0;

  // Model
  auto model_offset = CreateModel(_builder, version, operator_codes, subgraphs, description,
                                  buffers, 0 /* metadata_buffer */, metadata);
  FinishModelBuffer(_builder, model_offset);
}

const char *CircleExporterImpl::getBufferPointer() const
{
  return reinterpret_cast<const char *>(_builder.GetBufferPointer());
}

size_t CircleExporterImpl::getBufferSize() const { return _builder.GetSize(); }

} // namespace luci
