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
#include "CircleExportMetadata.h"
#include "CircleTensorExporter.h"
#include "CircleOperationExporter.h"
#include "CircleExporterUtils.h"
#include "ProgressReporter.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Pass/CircleShapeInferencePass.h>
#include <luci/Pass/CircleTypeInferencePass.h>

#include <loco.h>
#include <logo/Phase.h>
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
  for (const auto &it : opcodes)
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

namespace
{

void optimize(loco::Graph *g)
{
  logo::Phase phase;
  {
    // prepare type and shape before optimization
    phase.emplace_back(std::make_unique<luci::CircleShapeInferencePass>());
    phase.emplace_back(std::make_unique<luci::CircleTypeInferencePass>());

    // TODO add more optimization passes (with a knob)
  }

  logo::PhaseRunner<logo::PhaseStrategy::Restart> phase_runner{g};

  luci::ProgressReporter prog(g, logo::PhaseStrategy::Restart);
  phase_runner.attach(&prog);
  phase_runner.run(phase);
}

} // namespace

namespace luci
{

using namespace circle;
using namespace flatbuffers;

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

void CircleExporterImpl::exportModule(Module *module)
{
  assert(module->size() > 0);
  // do graph optimization

  SerializedModelData md;

  _builder.Clear();

  // prepare model data
  prepareModelData(_builder, md);

  // if source is extended buffer mode, force export to use extended buffer
  md._ext_buffer = module->ext_buffer();

  if (!exportModuleData(module, md) && md._require_ext_buffer)
  {
    assert(md._ext_buffer == false);

    // do some cleanups for re-run
    _builder.Clear();
    for (size_t g = 0; g < module->size(); ++g)
    {
      auto graph = module->graph(g);
      clearExportInfo(graph);
    }
    prepareModelData(_builder, md);

    // run again with ext_buffer mode
    md._ext_buffer = true;
    exportModuleData(module, md);
  }

  finalizeWithExtendedBuffer(md);
}

bool CircleExporterImpl::exportModuleData(Module *module, SerializedModelData &md)
{
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
  std::string description_str = "ONE-luci/export";
  auto description = _builder.CreateString(description_str);

  // Metadata
  md._metadata.source_table(module->source_table());
  auto metadata_vec = createCircleMetadataVector(_builder, md);
  auto metadata = _builder.CreateVector(std::vector<Offset<Metadata>>(metadata_vec));

  // create array of buffers
  auto buffers = _builder.CreateVector(md._buffers);

  // check current total size exceeds limit
  if (check_size_limit(_builder, 0))
  {
    md._require_ext_buffer = true;
    return false;
  }

  // This version is taken from comment in fbs
  constexpr uint32_t version = 0;

  // Model
  auto model_offset = CreateModel(_builder, version, operator_codes, subgraphs, description,
                                  buffers, 0 /* metadata_buffer */, metadata);
  FinishModelBuffer(_builder, model_offset);

  return true;
}

void CircleExporterImpl::finalizeWithExtendedBuffer(SerializedModelData &md)
{
  _ext_buffer = md._ext_buffer;
  if (!_ext_buffer)
    return;

  _fb_data_with_ext.clear();

  auto align16 = [](size_t &v) {
    while (v % 16 != 0)
      v++;
  };

  // get total memory for flatbuffer + all buffer_data
  size_t result_size = _builder.GetSize();
  align16(result_size);
  for (auto &it : md._buffer_data_map)
  {
    SerializedModelData::BufferData &buffer_data = it.second;
    result_size += buffer_data.size();
    align16(result_size);
  }
  align16(result_size);
  result_size += 16; // for safety

  std::string result;
  const char *buff_ptr = reinterpret_cast<const char *>(_builder.GetBufferPointer());

  auto padalign16 = [](std::string &str) {
    while (str.size() % 16 != 0)
      str += '\0';
  };

  result.reserve(result_size);
  result.append(buff_ptr, _builder.GetSize());

  auto mutable_model = circle::GetMutableModel(result.data());
  auto mutable_buffers = mutable_model->mutable_buffers();

  // pad to be 16 bytes aligned
  padalign16(result);
  for (auto &it : md._buffer_data_map)
  {
    int32_t buffer_index = it.first;
    SerializedModelData::BufferData &buffer_data = it.second;
    uint64_t offset = result.size();
    uint64_t size = buffer_data.size();

    circle::Buffer *mutable_buffer = mutable_buffers->GetMutableObject(buffer_index);
    mutable_buffer->mutate_offset(offset);
    mutable_buffer->mutate_size(size);

    result.append(buffer_data.begin(), buffer_data.end());
    padalign16(result);
  }
  padalign16(result);

  // use final result
  _fb_data_with_ext = result;
}

const char *CircleExporterImpl::getBufferPointer() const
{
  if (_ext_buffer)
    return reinterpret_cast<const char *>(_fb_data_with_ext.data());
  return reinterpret_cast<const char *>(_builder.GetBufferPointer());
}

size_t CircleExporterImpl::getBufferSize() const
{
  if (_ext_buffer)
    return _fb_data_with_ext.size();
  return _builder.GetSize();
}

} // namespace luci
