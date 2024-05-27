/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "MinMaxRecorder.h"

#include "backend/ITensor.h"

#include <cassert>
#include <cmath>

namespace onert
{
namespace exec
{

MinMaxRecorder::MinMaxRecorder(const std::string &minmax_filepath, const ir::Graph &graph,
                               const backend::BackendContexts &backend_contexts)
#if MINMAX_H5DUMPER
  : _graph{graph}, _backend_contexts{backend_contexts}, _h5dumper(minmax_filepath)
#else
  : _graph{graph}, _backend_contexts{backend_contexts}, _raw_dumper(minmax_filepath)
#endif
{
}

std::pair<float, float> minmaxFrom(const backend::ITensor *tensor)
{
  const auto data = reinterpret_cast<float *>(tensor->buffer());
  const auto num_elements = tensor->total_size() / sizeof(float);

  float max = std::numeric_limits<float>::lowest();
  float min = std::numeric_limits<float>::max();

  bool all_nan = true;
  for (size_t i = 0; i < num_elements; ++i)
  {
    const float number = data[i];
    if (std::isnan(number))
      continue;

    if (number == std::numeric_limits<float>::lowest())
      continue;

    all_nan = false;

    if (number > max)
      max = number;

    if (number < min)
      min = number;
  }

  if (all_nan)
    throw std::runtime_error("All values are NaN(Not a Number)");

  return {min, max};
}

void MinMaxRecorder::handleJobEnd(IExecutor *, ir::SubgraphIndex subg_idx,
                                  ir::OperationIndex op_idx, const backend::Backend *backend)
{
  const auto &tensor_reg = _backend_contexts.at(backend)->tensor_registry;
  const auto &op = _graph.operations().at(op_idx);
  const auto &outputs = op.getOutputs();
  // TODO: Support multiple output
  if (outputs.size() != 1)
    throw std::runtime_error("Only 1 output operator is supported for recording minmax.");

  auto tensor = tensor_reg->getITensor(outputs.at(0));

  // Logic copied from MinMaxObserver.cpp.

  // Filter Ops
  if (tensor->is_constant())
    return;

  if (tensor->data_type() != ir::DataType::FLOAT32)
    return;

  switch (op.opcode())
  {
    // Operators with multiple outputs
    case ir::OpCode::If:
    case ir::OpCode::Split:
    case ir::OpCode::SplitV:
    case ir::OpCode::TopKV2:
    case ir::OpCode::Unpack:
    case ir::OpCode::While:
      return;
    // NOTE: Sin, Cos, Tanh's output is in [-1, 1]
    // We may not need to dump those operators.
    default:; // Do Nothing
  }

  // Otherwise, dump!
  assert(tensor->data_type() == ir::DataType::FLOAT32);
  auto minmax = minmaxFrom(tensor);
  _op_minmax.append({subg_idx, op_idx}, minmax.first, minmax.second);
}

void MinMaxRecorder::handleSubgraphBegin(ir::SubgraphIndex subg_idx)
{
  // Make sure there is only cpu backend except for builtin backend
  std::set<std::string> backend_names;
  backend::ITensorRegistry *tensor_reg = nullptr;
  for (const auto &pair : _backend_contexts)
  {
    backend_names.insert(pair.first->config()->id());
    if (pair.first->config()->id() == "cpu")
    {
      tensor_reg = pair.second->tensor_registry.get();
    }
  }
  if (backend_names != std::set<std::string>{"builtin", "cpu"})
    throw std::runtime_error("MinMaxRecorder must have cpu backend only.");

  const auto &inputs = _graph.getInputs(); //.at(op_idx);
  for (uint32_t i = 0; i < inputs.size(); ++i)
  {
    auto input_idx = inputs.at(i);
    auto tensor = tensor_reg->getITensor(input_idx);

    if (tensor->is_constant())
      return;
    if (tensor->data_type() != ir::DataType::FLOAT32)
      return;

    auto minmax = minmaxFrom(tensor);
    _input_minmax.append({subg_idx, ir::IOIndex{i}}, minmax.first, minmax.second);
  }
}

void MinMaxRecorder::handleSubgraphEnd(ir::SubgraphIndex)
{
  // It would be better to dump at the end of model execution, not subgraph
  // But it requires more changes than subgraph.
#if MINMAX_H5DUMPER
  _h5dumper.dump(_input_minmax, _op_minmax);
#else
  _raw_dumper.dump(_input_minmax, _op_minmax);
#endif
}

} // namespace exec
} // namespace onert
