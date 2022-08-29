/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "exec/Executors.h"

namespace onert
{
namespace exec
{

uint32_t Executors::inputSize() const
{
  return _model_edges ? _model_edges->pkg_inputs.size()
                      : _executors.at(ir::SubgraphIndex{0})->graph().getInputs().size();
}

uint32_t Executors::outputSize() const
{
  return _model_edges ? _model_edges->pkg_outputs.size()
                      : _executors.at(ir::SubgraphIndex{0})->graph().getOutputs().size();
}

const ir::OperandInfo Executors::inputInfo(const ir::IOIndex &index)
{
  if (_model_edges)
  {
    // Assume that each model may have only one subgraph
    // TODO handle general case
    const auto desc = _model_edges->pkg_inputs[index.value()];
    const auto model_idx = std::get<0>(desc);
    const auto executor_idx = ir::SubgraphIndex{model_idx.value()};
    const auto input_index = _executors.at(executor_idx)->graph().getInputs().at(std::get<2>(desc));
    return _executors.at(executor_idx)->graph().operands().at(input_index).info();
  }

  const auto input_index = _executors.at(ir::SubgraphIndex{0})->graph().getInputs().at(index);
  return _executors.at(ir::SubgraphIndex{0})->graph().operands().at(input_index).info();
}

const ir::OperandInfo Executors::outputInfo(const ir::IOIndex &index)
{
  if (_model_edges)
  {
    // Assume that each model may have only one subgraph
    // TODO handle general case
    auto desc = _model_edges->pkg_outputs[index.value()];
    auto model_idx = std::get<0>(desc);
    auto executor_idx = ir::SubgraphIndex{model_idx.value()};
    auto output_index = _executors.at(executor_idx)->graph().getOutputs().at(std::get<2>(desc));
    return _executors.at(executor_idx)->graph().operands().at(output_index).info();
  }

  auto output_index = _executors.at(ir::SubgraphIndex{0})->graph().getOutputs().at(index);
  return _executors.at(ir::SubgraphIndex{0})->graph().operands().at(output_index).info();
}

void Executors::execute(const IODescription &desc)
{
  if (_model_edges)
    return executeEntries(desc);

  _executors.at(ir::SubgraphIndex{0})->execute(desc);
}

void Executors::executeEntries(const IODescription &desc)
{
  // Assume 2 executors only
  // Assume that each model may have only one subgraph
  // Assume that each model may have only one input/output
  // TODO Support general case
  if (_executors.size() != 2 || _model_edges->pkg_inputs.size() != 1 ||
      _model_edges->pkg_outputs.size() != 1 || _model_edges->edges.size() != 1)
    throw std::runtime_error{"NYI: Multi model execution for this package is not supported yet"};

  // Assume edge is 0:0:0 -> 1:0:0
  auto &edge = *_model_edges->edges.begin();
  if ((std::get<0>(edge.from) != ir::ModelIndex{0}) ||
      (std::get<1>(edge.from) != ir::SubgraphIndex{0}) ||
      (std::get<2>(edge.from) != ir::IOIndex{0}))
    throw std::runtime_error{"NYI: Multi model execution for this edge(from) is not supported yet"};

  if ((std::get<0>(edge.to) != ir::ModelIndex{1}) ||
      (std::get<1>(edge.to) != ir::SubgraphIndex{0}) || (std::get<2>(edge.to) != ir::IOIndex{0}))
    throw std::runtime_error{"NYI: Multi model execution for this edge(to) is not supported yet"};

  // Prepare buffer
  // Assume buffer layout is NHWC
  const auto layout = ir::Layout::NHWC;
  const auto buf_index =
    _executors.at(ir::SubgraphIndex{0})->graph().getOutputs().at(ir::IOIndex{0});
  const auto buf_info =
    _executors.at(ir::SubgraphIndex{0})->graph().operands().at(buf_index).info();
  const auto buf_size = buf_info.total_size();
  auto connect_buf = std::make_unique<uint8_t[]>(buf_size);
  auto buf_ptr = connect_buf.get();

  // 1st executor
  {
    auto &executor1 = _executors.at(ir::SubgraphIndex{0});
    auto &graph1 = executor1->graph();
    if (graph1.getInputs().size() != 1 || graph1.getOutputs().size() != 1)
      throw std::runtime_error{
        "NYI: Multi model execution for this 1st model is not supported yet"};

    const auto input_desc = _model_edges->pkg_inputs[0];
    if ((std::get<0>(input_desc) != ir::ModelIndex{0}) ||
        (std::get<1>(input_desc) != ir::SubgraphIndex{0}) ||
        (std::get<2>(input_desc) != ir::IOIndex{0}))
      throw std::runtime_error{
        "NYI: Multi model execution for this 1st model is not supported yet"};

    IODescription desc1;
    desc1.inputs.resize(1);
    desc1.inputs[0] = std::make_unique<InputDesc>(*desc.inputs[0].get());
    desc1.outputs.resize(1);
    desc1.outputs[0] = std::make_unique<OutputDesc>(buf_info, buf_ptr, buf_size, layout);
    executor1->execute(desc1);
  }

  // 2nd executor
  {
    auto &executor2 = _executors.at(ir::SubgraphIndex{1});
    auto &graph2 = executor2->graph();
    if (graph2.getInputs().size() != 1 || graph2.getOutputs().size() != 1)
      throw std::runtime_error{
        "NYI: Multi model execution for this 2nd model is not supported yet"};

    const auto output_desc = _model_edges->pkg_outputs[0];
    if ((std::get<0>(output_desc) != ir::ModelIndex{1}) ||
        (std::get<1>(output_desc) != ir::SubgraphIndex{0}) ||
        (std::get<2>(output_desc) != ir::IOIndex{0}))
      throw std::runtime_error{
        "NYI: Multi model execution for this 2nd model is not supported yet"};

    IODescription desc2;
    desc2.inputs.resize(1);
    desc2.inputs[0] = std::make_unique<InputDesc>(buf_info, buf_ptr, buf_size, layout);
    desc2.outputs.resize(1);
    desc2.outputs[0] = std::make_unique<OutputDesc>(*desc.outputs[0].get());
    executor2->execute(desc2);
  }
}

} // namespace exec
} // namespace onert
