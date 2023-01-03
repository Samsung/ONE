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

#include "Executors.h"

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

// Allow below case only
//  input(s) -> 1st model -> 2nd model -> ... -> (n-1)th model -> (n)th model -> output(s)
void Executors::checkSupportedMultimodel() const
{
  auto const model_count = modelCount();

  // If package includes no-connection model, model_count is less than real model count in package.
  // Then this method will throw exception based on model index
  //  1st model: input assumption
  //  (n)th model: output assumption
  //  Otherwise: edges assumption

  // Assumption: edges
  //   x:0:z -> x+1:0:z'
  for (auto edge : _model_edges->edges)
  {
    auto const model_from = std::get<ir::ModelIndex>(edge.from);
    auto const model_to = std::get<ir::ModelIndex>(edge.to);
    auto const subg_from = std::get<ir::SubgraphIndex>(edge.from);
    auto const subg_to = std::get<ir::SubgraphIndex>(edge.to);

    if (((model_from.value() + 1) != model_to.value()) || (subg_from != ir::SubgraphIndex{0}) ||
        (subg_to != ir::SubgraphIndex{0}))
      throw std::runtime_error{"NYI: Multi model execution for this edge set is not supported yet"};
  }

  // Assumption: package inputs
  //  All 1st model inputs come from package input
  {
    auto first_executor = at(ir::ModelIndex{0}, ir::SubgraphIndex{0});
    auto search_first_model = [&](const ir::IOIndex &input_index) {
      for (auto &input : _model_edges->pkg_inputs)
      {
        if ((std::get<ir::ModelIndex>(input) == ir::ModelIndex{0}) ||
            (std::get<ir::SubgraphIndex>(input) == ir::SubgraphIndex{0}) ||
            (std::get<ir::IOIndex>(input) == input_index))
          return true;
      }

      return false;
    };

    for (uint32_t i = 0; i < first_executor->graph().getInputs().size(); i++)
    {
      if (!search_first_model(ir::IOIndex{i}))
        throw std::runtime_error{"Cannot find 1st model's input buffer"};
    }
  }

  // Assumption: package outputs
  //  All last model outputs are go to package output
  {
    auto last_model_index = ir::ModelIndex{static_cast<uint16_t>(model_count - 1)};
    auto last_executor = at(last_model_index, ir::SubgraphIndex{0});
    auto search_last_model = [&](const ir::IOIndex &output_index) {
      for (auto &output : _model_edges->pkg_outputs)
      {
        if ((std::get<ir::ModelIndex>(output) == last_model_index) ||
            (std::get<ir::SubgraphIndex>(output) == ir::SubgraphIndex{0}) ||
            (std::get<ir::IOIndex>(output) == output_index))
          return true;
      }

      return false;
    };

    for (uint32_t i = 0; i < last_executor->graph().getOutputs().size(); i++)
    {
      if (!search_last_model(ir::IOIndex{i}))
        throw std::runtime_error{"Cannot find last model's output buffer"};
    }
  }
}

void Executors::executeModels(const IODescription &desc)
{
  // Check supported multi model package
  checkSupportedMultimodel();

  // TODO Find better way to manage buffer between executors
  std::vector<std::unique_ptr<uint8_t[]>> prev_bufs;
  std::vector<std::unique_ptr<uint8_t[]>> curr_bufs;
  const auto layout = ir::Layout::NHWC;
  auto const model_count = modelCount();

  auto find_input_index = [&](const ir::ModelIndex &model_index,
                              const ir::SubgraphIndex &subg_index, const ir::IOIndex &io_index) {
    for (size_t i = 0; i < _model_edges->pkg_inputs.size(); i++)
    {
      auto &input_desc = _model_edges->pkg_inputs[i];
      if ((std::get<ir::ModelIndex>(input_desc) == model_index) &&
          (std::get<ir::SubgraphIndex>(input_desc) == subg_index) &&
          (std::get<ir::IOIndex>(input_desc) == io_index))
        return static_cast<int32_t>(i);
    }
    return -1;
  };

  auto find_output_index = [&](const ir::ModelIndex &model_index,
                               const ir::SubgraphIndex &subg_index, const ir::IOIndex &io_index) {
    for (size_t i = 0; i < _model_edges->pkg_outputs.size(); i++)
    {
      auto &input_desc = _model_edges->pkg_outputs[i];
      if ((std::get<ir::ModelIndex>(input_desc) == model_index) &&
          (std::get<ir::SubgraphIndex>(input_desc) == subg_index) &&
          (std::get<ir::IOIndex>(input_desc) == io_index))
        return static_cast<int32_t>(i);
    }
    return -1;
  };

  auto find_from = [&](const ir::ModelIndex &model_index, const ir::SubgraphIndex &subg_index,
                       const ir::IOIndex &io_index) {
    for (auto &edge : _model_edges->edges)
    {
      if ((std::get<ir::ModelIndex>(edge.to) == model_index) &&
          (std::get<ir::SubgraphIndex>(edge.to) == subg_index) &&
          (std::get<ir::IOIndex>(edge.to) == io_index))
        return edge.from;
    }

    throw std::runtime_error{"Cannot find edge for model input"};
  };

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

    for (uint32_t i = 0; i < input_size; i++)
    {
      auto const &index = executor->graph().getInputs().at(ir::IOIndex{i});
      auto const &info = executor->graph().operands().at(index).info();

      auto input_pkg_index = find_input_index(model_index, ir::SubgraphIndex{0}, ir::IOIndex{i});
      if (input_pkg_index != -1)
      {
        desc_inter.inputs[i] = std::make_unique<InputDesc>(*desc.inputs[input_pkg_index].get());
        continue;
      }

      auto from_iodesc = find_from(model_index, ir::SubgraphIndex{0}, ir::IOIndex{i});
      auto from_ioindex = std::get<ir::IOIndex>(from_iodesc).value();

      // Comes from previous model
      assert(std::get<ir::ModelIndex>(from_iodesc).value() + 1 == model_index.value());
      assert(std::get<ir::SubgraphIndex>(from_iodesc).value() == 0);

      desc_inter.inputs[i] =
        std::make_unique<InputDesc>(info, prev_bufs[from_ioindex].get(), info.total_size(), layout);
    }

    curr_bufs.resize(output_size);
    for (uint32_t i = 0; i < output_size; i++)
    {
      auto const &index = executor->graph().getOutputs().at(ir::IOIndex{i});
      auto const &info = executor->graph().operands().at(index).info();

      auto output_pkg_index = find_output_index(model_index, ir::SubgraphIndex{0}, ir::IOIndex{i});
      if (output_pkg_index != -1)
      {
        desc_inter.outputs[i] = std::make_unique<OutputDesc>(*desc.outputs[output_pkg_index].get());
        continue;
      }

      curr_bufs[i] = std::make_unique<uint8_t[]>(info.total_size());
      desc_inter.outputs[i] =
        std::make_unique<OutputDesc>(info, curr_bufs[i].get(), info.total_size(), layout);
    }

    executor->execute(desc_inter);

    if (model_index.value() + 1 != model_count)
    {
      // Backup output buffer to input buffer for next execution
      prev_bufs.resize(output_size);
      for (uint32_t i = 0; i < output_size; i++)
        prev_bufs[i] = std::move(curr_bufs[i]);
    }
  }
}

// modelCount() iterates _executors.
// It assumes that Compiler will generate Executor for all models and _executors includes all
// generated Executor.
// If nnpackage includes model(s) which has no connection and Compiler does not
// generate Executor for them, modelCount() return less value than real model count.
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
