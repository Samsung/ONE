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

void Executors::checkSupportedMultimodel() const
{
  // Assumption
  // Models
  //   size: n
  //   sizeof(model(a).outputs) == sizeof(model(a+1).inputs)
  auto const model_count = modelCount();

  for (auto &pair : _executors)
  {
    auto const &model_index = pair.first.first;
    auto const &subg_index = pair.first.second;

    // Inner subgraph
    if (subg_index != ir::SubgraphIndex{0})
      continue;

    // Last model's output
    auto const next_index = ir::ModelIndex{static_cast<uint16_t>(model_index.value() + 1)};
    if (_executors.find(std::make_pair(next_index, ir::SubgraphIndex{0})) == _executors.end())
    {
      auto executor = at(model_index, subg_index);
      if (executor->graph().getOutputs().size() != _model_edges->pkg_outputs.size())
        throw std::runtime_error{"NYI: Unsupported model edge pattern"};

      continue;
    }

    auto executor_from = at(model_index, subg_index);
    auto executor_to = at(next_index, subg_index);

    if (executor_from->graph().getOutputs().size() != executor_to->graph().getInputs().size())
      throw std::runtime_error{"NYI: Multi model execution for this package is not supported yet"};

    // 1st model's input
    if ((model_index == ir::ModelIndex{0}) &&
        (executor_from->graph().getInputs().size() != _model_edges->pkg_inputs.size()))
      throw std::runtime_error{"NYI: Unsupported model edge pattern"};
  }

  // Edges
  //   a:0:z -> a+1:0:z  (0 < a < n-2)
  for (auto edge : _model_edges->edges)
  {
    auto const model_from = std::get<ir::ModelIndex>(edge.from);
    auto const model_to = std::get<ir::ModelIndex>(edge.to);
    auto const subg_from = std::get<ir::SubgraphIndex>(edge.from);
    auto const subg_to = std::get<ir::SubgraphIndex>(edge.to);
    auto const output_from = std::get<ir::IOIndex>(edge.from);
    auto const input_to = std::get<ir::IOIndex>(edge.to);

    if (((model_from.value() + 1) != model_to.value()) || (subg_from != ir::SubgraphIndex{0}) ||
        (subg_to != ir::SubgraphIndex{0}) || (output_from != input_to))
      throw std::runtime_error{"NYI: Multi model execution for this edge set is not supported yet"};
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

  // Assume all package outputs are n-1:0:x
  for (uint32_t i = 0; i < _model_edges->pkg_outputs.size(); i++)
  {
    auto output = _model_edges->pkg_outputs[i];
    if ((std::get<ir::ModelIndex>(output) !=
         ir::ModelIndex{static_cast<uint16_t>(model_count - 1)}) ||
        (std::get<ir::SubgraphIndex>(output) != ir::SubgraphIndex{0}) ||
        (std::get<ir::IOIndex>(output) != ir::IOIndex{i}))
    {
      throw std::runtime_error{"NYI: Support package output from (n-1)th model with same order"};
    }
  }
}

void Executors::executeModels(const IODescription &desc)
{
  // Check supported multi model package
  checkSupportedMultimodel();

  // TODO Find better way to manage buffer between executors
  std::vector<std::unique_ptr<uint8_t[]>> input_bufs;
  std::vector<std::unique_ptr<uint8_t[]>> output_bufs;
  const auto layout = ir::Layout::NHWC;
  auto const model_count = modelCount();

  // Execute each model
  // NOTE May be better to use vector instead of unordered_map for _executors
  for (auto model_index = ir::ModelIndex{0}; model_index.value() < model_count; model_index++)
  {
    // Find executor
    auto executor = at(model_index, ir::SubgraphIndex{0});

    // Set IODescription
    IODescription desc_inter;
    auto const input_size = executor->graph().getInputs().size();
    auto const output_size = executor->graph().getOutputs().size();
    desc_inter.inputs.resize(input_size);
    desc_inter.outputs.resize(output_size);

    input_bufs.resize(input_size);
    for (uint32_t i = 0; i < input_size; i++)
    {
      auto const &index = executor->graph().getInputs().at(ir::IOIndex{i});
      auto const &info = executor->graph().operands().at(index).info();

      // 1st model
      if (model_index == 0)
      {
        assert(desc.inputs[i]->info.total_size() == info.total_size());
        desc_inter.inputs[i] = std::make_unique<InputDesc>(*desc.inputs[i].get());
        continue;
      }

      input_bufs[i] = std::move(output_bufs[i]);
      desc_inter.inputs[i] =
        std::make_unique<InputDesc>(info, input_bufs[i].get(), info.total_size(), layout);
    }

    output_bufs.resize(output_size);
    for (uint32_t i = 0; i < output_size; i++)
    {
      auto const &index = executor->graph().getOutputs().at(ir::IOIndex{i});
      auto const &info = executor->graph().operands().at(index).info();

      // Last model
      if (model_index.value() + 1 == model_count)
      {
        assert(desc.outputs[i]->info.total_size() == info.total_size());
        desc_inter.outputs[i] = std::make_unique<OutputDesc>(*desc.outputs[i].get());
        continue;
      }

      output_bufs[i] = std::make_unique<uint8_t[]>(info.total_size());
      desc_inter.outputs[i] =
        std::make_unique<OutputDesc>(info, output_bufs[i].get(), info.total_size(), layout);
    }

    executor->execute(desc_inter);
  }
}

uint16_t Executors::modelCount() const
{
  uint16_t model_count = 0;
  for (; _executors.find(std::make_pair(ir::ModelIndex{model_count}, ir::SubgraphIndex{0})) !=
         _executors.end();
       model_count++)
    ;

  return model_count;
}

} // namespace exec
} // namespace onert
