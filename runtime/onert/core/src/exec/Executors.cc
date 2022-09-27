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

void Executors::emplace(const ir::ModelIndex &model_index, const ir::SubgraphIndex &subg_index,
                        std::unique_ptr<IExecutor> exec)
{
  _executors.emplace(std::make_pair(model_index, subg_index), std::move(exec));
}

IExecutor *Executors::at(const ir::ModelIndex &model_index,
                         const ir::SubgraphIndex &subg_index) const
{
  return _executors.at(std::make_pair(model_index, subg_index)).get();
}

uint32_t Executors::inputSize() const
{
  return _model_edges ? _model_edges->pkg_inputs.size()
                      : entryExecutor()->graph().getInputs().size();
}

uint32_t Executors::outputSize() const
{
  return _model_edges ? _model_edges->pkg_outputs.size()
                      : entryExecutor()->graph().getOutputs().size();
}

const ir::OperandInfo Executors::inputInfo(const ir::IOIndex &index)
{
  if (_model_edges)
  {
    auto const desc = _model_edges->pkg_inputs[index.value()];
    auto const model_index = std::get<0>(desc);
    auto const subg_index = std::get<1>(desc);
    auto const executor = at(model_index, subg_index);
    auto const input_index = executor->graph().getInputs().at(std::get<2>(desc));
    return executor->graph().operands().at(input_index).info();
  }

  const auto input_index = entryExecutor()->graph().getInputs().at(index);
  return entryExecutor()->graph().operands().at(input_index).info();
}

const ir::OperandInfo Executors::outputInfo(const ir::IOIndex &index)
{
  if (_model_edges)
  {
    auto const desc = _model_edges->pkg_outputs[index.value()];
    auto const model_index = std::get<0>(desc);
    auto const subg_index = std::get<1>(desc);
    auto const executor = at(model_index, subg_index);
    auto const output_index = executor->graph().getOutputs().at(std::get<2>(desc));
    return executor->graph().operands().at(output_index).info();
  }

  auto output_index = entryExecutor()->graph().getOutputs().at(index);
  return entryExecutor()->graph().operands().at(output_index).info();
}

void Executors::execute(const IODescription &desc)
{
  if (_model_edges)
    return executeModels(desc);

  entryExecutor()->execute(desc);
}

void Executors::executeModels(const IODescription &desc)
{
  // Assume 2 executors only
  // Assume that each model may have only one subgraph
  // TODO Support general case
  if (_executors.size() != 2)
    throw std::runtime_error{"NYI: Multi model execution for this package is not supported yet"};

  // Assume all edges are 0:0:x -> 1:0:x
  for (auto edge : _model_edges->edges)
  {
    if ((std::get<ir::ModelIndex>(edge.from) != ir::ModelIndex{0}) ||
        (std::get<ir::ModelIndex>(edge.to) != ir::ModelIndex{1}) ||
        (std::get<ir::SubgraphIndex>(edge.from) != ir::SubgraphIndex{0}) ||
        (std::get<ir::SubgraphIndex>(edge.to) != ir::SubgraphIndex{0}) ||
        (std::get<ir::IOIndex>(edge.from) != std::get<ir::IOIndex>(edge.to)))
      throw std::runtime_error{"NYI: Multi model execution for this edge is not supported yet"};
  }

  // Assume all package inputs are 0:0:x
  for (uint32_t i = 0; i < _model_edges->pkg_inputs.size(); i++)
  {
    auto input = _model_edges->pkg_inputs[i];
    if ((std::get<ir::ModelIndex>(input) != ir::ModelIndex{0}) ||
        (std::get<ir::SubgraphIndex>(input) != ir::SubgraphIndex{0}) ||
        (std::get<ir::IOIndex>(input) != ir::IOIndex{i}))
    {
      throw std::runtime_error{"NYI: Support package input to 1st model with same order"};
    }
  }

  // Assume all package outputs are 1:0:x
  for (uint32_t i = 0; i < _model_edges->pkg_outputs.size(); i++)
  {
    auto output = _model_edges->pkg_outputs[i];
    if ((std::get<ir::ModelIndex>(output) != ir::ModelIndex{1}) ||
        (std::get<ir::SubgraphIndex>(output) != ir::SubgraphIndex{0}) ||
        (std::get<ir::IOIndex>(output) != ir::IOIndex{i}))
    {
      throw std::runtime_error{"NYI: Support package output from 2nd model with same order"};
    }
  }

  const auto &executor1 = at(ir::ModelIndex{0}, ir::SubgraphIndex{0});
  const auto &graph1 = executor1->graph();
  const auto &executor2 = at(ir::ModelIndex{1}, ir::SubgraphIndex{0});
  const auto &graph2 = executor2->graph();

  if ((graph1.getInputs().size() != _model_edges->pkg_inputs.size()) ||
      (graph2.getOutputs().size() != _model_edges->pkg_outputs.size()) ||
      (graph1.getOutputs().size() != graph2.getInputs().size()) ||
      (graph1.getOutputs().size() != _model_edges->edges.size()))
  {
    throw std::runtime_error{"NYI: Unsupported model edge pattern"};
  }

  // Prepare buffer
  // Assume buffer layout is NHWC
  std::vector<std::unique_ptr<uint8_t[]>> bufs(_model_edges->edges.size());
  std::vector<const ir::OperandInfo *> buf_infos(_model_edges->edges.size());
  const auto layout = ir::Layout::NHWC;

  for (uint32_t i = 0; i < graph1.getOutputs().size(); i++)
  {
    const auto buf_index = executor1->graph().getOutputs().at(ir::IOIndex{i});
    buf_infos[i] = &executor1->graph().operands().at(buf_index).info();
    const auto buf_size = buf_infos[i]->total_size();
    bufs[i] = std::make_unique<uint8_t[]>(buf_size);
  }

  // 1st executor
  {
    IODescription desc1;
    const auto input_size = graph1.getInputs().size();
    const auto output_size = graph1.getOutputs().size();
    desc1.inputs.resize(input_size);
    desc1.outputs.resize(output_size);
    for (uint32_t i = 0; i < input_size; i++)
      desc1.inputs[i] = std::make_unique<InputDesc>(*desc.inputs[i].get());
    for (uint32_t i = 0; i < output_size; i++)
      desc1.outputs[i] = std::make_unique<OutputDesc>(*buf_infos[i], bufs[i].get(),
                                                      buf_infos[i]->total_size(), layout);

    executor1->execute(desc1);
  }

  // 2nd executor
  {
    IODescription desc2;
    const auto input_size = graph2.getInputs().size();
    const auto output_size = graph2.getOutputs().size();
    desc2.inputs.resize(input_size);
    desc2.outputs.resize(output_size);
    for (uint32_t i = 0; i < input_size; i++)
      desc2.inputs[i] = std::make_unique<InputDesc>(*buf_infos[i], bufs[i].get(),
                                                    buf_infos[i]->total_size(), layout);
    for (uint32_t i = 0; i < output_size; i++)
      desc2.outputs[i] = std::make_unique<OutputDesc>(*desc.outputs[i].get());

    executor2->execute(desc2);
  }
}

} // namespace exec
} // namespace onert
