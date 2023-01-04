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

#include "../backend/builtin/IOTensor.h"

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

// Allow below edges only
//  m1 < m2, s1 == 0 and s2 == 0 if m1:s1:o1 -> m2:s2:o2'
void Executors::checkSupportedMultimodel() const
{
  // If package includes no-connection model, model_count is less than real model count in package.
  // Then this method will throw exception based on model index
  //  1st model: input assumption
  //  Otherwise: edges assumption

  // Assumption: edges
  // m1 < m2, s1 == 0 and s2 == 0 if edge 'm1:s1:o1 -> m2:s2:o2'
  for (auto edge : _model_edges->edges)
  {
    auto const model_from = std::get<ir::ModelIndex>(edge.from);
    auto const model_to = std::get<ir::ModelIndex>(edge.to);
    auto const subg_from = std::get<ir::SubgraphIndex>(edge.from);
    auto const subg_to = std::get<ir::SubgraphIndex>(edge.to);

    if (model_from.value() == model_to.value())
    {
      throw std::runtime_error{"Multi model's edge set has invalid edge"};
    }

    if ((model_from.value() > model_to.value()) || (subg_from != ir::SubgraphIndex{0}) ||
        (subg_to != ir::SubgraphIndex{0}))
      throw std::runtime_error{"NYI: Multi model execution for this edge set is not supported yet"};
  }

  // Assumption: package inputs
  //  All 1st model inputs come from package input if always m1 < m2
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
}

void Executors::executeModels(const IODescription &desc)
{
  // Check supported multi model package
  checkSupportedMultimodel();

  // TODO Find better way to schedule order of executors
  std::vector<std::unique_ptr<backend::builtin::IOTensor>> pkgs_inputs(desc.inputs.size());
  std::vector<std::unique_ptr<backend::builtin::IOTensor>> pkgs_outputs(desc.outputs.size());
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

  // TODO Find better way to share buffers and tensors between executors
  // NOTE Executor's inputs/outputs are always IPortableTensor. If backend of inputs/outputs
  //      is using tensor that does not inherit IPortableTensor, Permute operation is added
  //      and all inputs/outputs become IPortableTensor at compile stage.
  //      This allows user's buffers to be set to inputs/outputs of executors.
  // Shared buffers and tensors between executors
  std::vector<std::unique_ptr<uint8_t[]>> edge_bufs;
  std::vector<std::unique_ptr<backend::builtin::IOTensor>> edge_tensor;

  // Execute each model
  // NOTE May be better to use vector instead of unordered_map for _executors
  for (auto model_index = ir::ModelIndex{0}; model_index.value() < model_count; model_index++)
  {
    // Find executor
    auto executor = at(model_index, ir::SubgraphIndex{0});

    // Set IOTensors
    std::vector<backend::IPortableTensor *> inputs_inter;
    std::vector<backend::IPortableTensor *> outputs_inter;
    auto const input_size = executor->graph().getInputs().size();
    auto const output_size = executor->graph().getOutputs().size();
    inputs_inter.resize(input_size);
    outputs_inter.resize(output_size);

    // Set inputs of executor
    for (uint32_t i = 0; i < input_size; i++)
    {
      auto const &index = executor->graph().getInputs().at(ir::IOIndex{i});
      auto const &info = executor->graph().operands().at(index).info();

      auto input_pkg_index = find_input_index(model_index, ir::SubgraphIndex{0}, ir::IOIndex{i});
      if (input_pkg_index != -1)
      {
        auto input_desc = desc.inputs[input_pkg_index].get();
        pkgs_inputs[input_pkg_index] = std::make_unique<backend::builtin::IOTensor>(info, layout);
        // TODO Remove const_cast (we need const_cast as ITensor is writable)
        pkgs_inputs[input_pkg_index]->setUserTensor(
          reinterpret_cast<uint8_t *>(const_cast<void *>(input_desc->buffer)), input_desc->size);

        inputs_inter[i] = pkgs_inputs[input_pkg_index].get();
      }
      else
      {
        auto from_iodesc = find_from(model_index, ir::SubgraphIndex{0}, ir::IOIndex{i});
        const auto &from_model_index = std::get<ir::ModelIndex>(from_iodesc);
        const auto &from_subg_index = std::get<ir::SubgraphIndex>(from_iodesc);
        const auto &from_ioindex = std::get<ir::IOIndex>(from_iodesc).value();

        // Supported only sequantial execution of models
        assert(from_model_index.value() < model_index.value());
        assert(from_subg_index.value() == 0);

        // TODO Add check if from_executor has already been executed
        const auto from_executor = _executors.at({from_model_index, from_subg_index}).get();
        inputs_inter[i] = from_executor->getOutputTensors().at(from_ioindex);
      }
    }

    // Set outputs of executor
    for (uint32_t i = 0; i < output_size; i++)
    {
      auto const &index = executor->graph().getOutputs().at(ir::IOIndex{i});
      auto const &info = executor->graph().operands().at(index).info();

      auto output_pkg_index = find_output_index(model_index, ir::SubgraphIndex{0}, ir::IOIndex{i});
      if (output_pkg_index != -1)
      {
        auto output_desc = desc.outputs[output_pkg_index].get();
        pkgs_outputs[output_pkg_index] = std::make_unique<backend::builtin::IOTensor>(info, layout);
        pkgs_outputs[output_pkg_index]->setUserTensor(
          reinterpret_cast<uint8_t *>(output_desc->buffer), output_desc->size);

        outputs_inter[i] = pkgs_outputs[output_pkg_index].get();
      }
      else
      {
        assert(edge_tensor.size() == edge_bufs.size());
        const auto temp_index = edge_tensor.size();
        edge_bufs.emplace_back(std::make_unique<uint8_t[]>(info.total_size()));
        edge_tensor.emplace_back(std::make_unique<backend::builtin::IOTensor>(info, layout));
        edge_tensor.at(temp_index)->setUserTensor(edge_bufs[temp_index].get(), info.total_size());
        outputs_inter[i] = edge_tensor[temp_index].get();
      }
    }

    executor->execute(inputs_inter, outputs_inter);
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
