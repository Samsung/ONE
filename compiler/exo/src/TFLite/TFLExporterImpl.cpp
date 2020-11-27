/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "TFLExporterImpl.h"

#include "Convert.h"
#include "ExoOptimize.h"

#include "TFLTensorExporter.h"
#include "TFLOperationExporter.h"
#include "TFLExporterUtils.h"

#include "Log.h"
#include "Knob.h"

#include <oops/InternalExn.h>

#include <cassert>
#include <unordered_map>
#include <string>
#include <stdexcept>

namespace
{

using namespace exo;
using namespace exo::tflite_detail;

void registerGraphInputTensors(loco::Graph *graph, SubGraphContext &ctx)
{
  for (uint32_t n = 0; n < graph->inputs()->size(); ++n)
  {
    auto node = loco::pull_node(graph, n);
    assert(node != nullptr);
    ctx._inputs.push_back(get_tensor_index(node));
  }
}

void registerGraphOutputTensors(loco::Graph *graph, SubGraphContext &ctx)
{
  for (uint32_t n = 0; n < graph->outputs()->size(); ++n)
  {
    auto push = loco::push_node(graph, n);
    assert(push != nullptr);
    auto node = push->from();
    assert(node != nullptr);
    ctx._outputs.push_back(get_tensor_index(node));
  }
}

} // namespace

namespace
{
using namespace tflite;
using namespace flatbuffers;

Offset<Vector<Offset<OperatorCode>>>
encodeOperatorCodes(FlatBufferBuilder &builder, std::unordered_map<OpCode, uint32_t> &opcodes,
                    std::unordered_map<OpCode, std::string> &custom_opcodes)
{
  std::vector<Offset<OperatorCode>> operator_codes_vec(opcodes.size());
  for (auto it : opcodes)
  {
    uint32_t idx = it.second;
    if (it.first.opcode != BuiltinOperator_CUSTOM)
    {
      operator_codes_vec[idx] = CreateOperatorCode(builder, it.first.opcode);
    }
    else // custom op
    {
      auto opCode = it.first;
      auto custom_code = custom_opcodes.find(opCode);
      if (custom_code == custom_opcodes.end())
        INTERNAL_EXN("Cannot find code for custom op");

      operator_codes_vec[idx] =
        CreateOperatorCode(builder, it.first.opcode, builder.CreateString(custom_code->second));
    }
  }
  return builder.CreateVector(operator_codes_vec);
}

} // namespace

namespace exo
{

using namespace exo::tflite_detail;
using namespace tflite;
using namespace flatbuffers;

TFLExporter::Impl::Impl(loco::Graph *graph) { exportGraph(graph); }

::flatbuffers::Offset<::tflite::SubGraph> TFLExporter::Impl::exportSubgraph(SerializedModelData &gd)
{
  auto tensors = _builder.CreateVector(gd._tensors);
  auto inputs = _builder.CreateVector(gd._inputs);
  auto outputs = _builder.CreateVector(gd._outputs);
  auto operators = _builder.CreateVector(gd._operators);
  auto subgraph = CreateSubGraph(_builder, tensors, inputs, outputs, operators);
  return subgraph;
}

void TFLExporter::Impl::exportGraph(loco::Graph *graph)
{
  LOGGER(l);

  // IR-level conversion and optimization
  {
    convert_to_TFLNodes(graph);
    set(Dialect::TFLITE);
    optimize(graph);
  }

  _builder.Clear();

  SerializedModelData gd;

  // This version is taken from comment in fbs
  constexpr uint32_t version = 3;

  registerGraphIOName(graph, gd);

  // parse graph into SerializedModelData structure
  exportOpDefinedTensors(graph, _builder, gd);

  // NOTE Invoke these register functions only after each node is annotated with its tensor_index
  registerGraphInputTensors(graph, gd);
  registerGraphOutputTensors(graph, gd);

  exportNodes(graph, _builder, gd);

  // encode operator codes
  auto operator_codes =
    encodeOperatorCodes(_builder, gd._operator_codes, gd._custom_operator_codes);

  // Subgraphs
  Offset<SubGraph> subgraph = exportSubgraph(gd);
  auto subgraphs = _builder.CreateVector(std::vector<Offset<SubGraph>>{subgraph});

  // Description
  std::string description_str = "nnpackage";
  auto description = _builder.CreateString(description_str);

  // create array of buffers
  auto buffers = _builder.CreateVector(gd._buffers);

  // empty metadata
  std::vector<int> metadata_buffer_vec;
  auto metadata_buffer = _builder.CreateVector(metadata_buffer_vec);

  // Model
  auto model_offset = CreateModel(_builder, version, operator_codes, subgraphs, description,
                                  buffers, metadata_buffer);
  FinishModelBuffer(_builder, model_offset);
}

const char *TFLExporter::Impl::getBufferPointer() const
{
  return reinterpret_cast<const char *>(_builder.GetBufferPointer());
}

size_t TFLExporter::Impl::getBufferSize() const { return _builder.GetSize(); }

} // namespace exo
