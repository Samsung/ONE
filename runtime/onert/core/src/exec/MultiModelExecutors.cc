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

#include "MultiModelExecutors.h"

#include "../backend/builtin/IOTensor.h"

namespace
{

using namespace onert;

int32_t find_input_index(const std::vector<ir::IODesc> &pkg_inputs,
                         const ir::ModelIndex &model_index, const ir::SubgraphIndex &subg_index,
                         const ir::IOIndex &io_index)
{
  for (size_t i = 0; i < pkg_inputs.size(); i++)
  {
    const auto &[m, s, io] = pkg_inputs[i];
    if ((m == model_index) && (s == subg_index) && (io == io_index))
      return static_cast<int32_t>(i);
  }
  return -1;
}

int32_t find_output_index(const std::vector<ir::IODesc> &pkg_outputs,
                          const ir::ModelIndex &model_index, const ir::SubgraphIndex &subg_index,
                          const ir::IOIndex &io_index)
{
  for (size_t i = 0; i < pkg_outputs.size(); i++)
  {
    const auto &[m, s, io] = pkg_outputs[i];
    if ((m == model_index) && (s == subg_index) && (io == io_index))
      return static_cast<int32_t>(i);
  }
  return -1;
}

} // namespace

namespace onert::exec
{

void MultiModelExecutors::emplace(const ir::ModelIndex &model_index,
                                  const ir::SubgraphIndex &subg_index,
                                  std::unique_ptr<IExecutor> exec)
{
  _executors.emplace(std::make_pair(model_index, subg_index), std::move(exec));
}

IExecutor *MultiModelExecutors::at(const ir::ModelIndex &model_index,
                                   const ir::SubgraphIndex &subg_index) const
{
  return _executors.at(std::make_pair(model_index, subg_index)).get();
}

uint32_t MultiModelExecutors::inputSize() const { return _model_edges->pkg_inputs.size(); }

uint32_t MultiModelExecutors::outputSize() const { return _model_edges->pkg_outputs.size(); }

const ir::OperandInfo &MultiModelExecutors::inputInfo(const ir::IOIndex &index) const
{
  auto const [model_index, subg_index, io_index] = _model_edges->pkg_inputs[index.value()];
  auto const executor = at(model_index, subg_index);
  return executor->inputInfo(io_index.value());
}

const ir::OperandInfo &MultiModelExecutors::outputInfo(const ir::IOIndex &index) const
{
  auto const [model_index, subg_index, io_index] = _model_edges->pkg_outputs[index.value()];
  auto const executor = at(model_index, subg_index);
  return executor->outputInfo(io_index.value());
}

const void *MultiModelExecutors::outputBuffer(const ir::IOIndex &index) const
{
  auto const [model_index, subg_index, io_index] = _model_edges->pkg_outputs[index.value()];
  auto const executor = at(model_index, subg_index);
  return static_cast<const void *>(executor->outputBuffer(io_index.value()));
}

const backend::IPortableTensor *MultiModelExecutors::outputTensor(const ir::IOIndex &index) const
{
  auto const [model_index, subg_index, io_index] = _model_edges->pkg_outputs[index.value()];
  auto const executor = at(model_index, subg_index);
  return executor->outputTensor(io_index.value());
}

// Allow below edges only
//  m1 < m2, s1 == 0 and s2 == 0 if m1:s1:o1 -> m2:s2:o2'
void MultiModelExecutors::checkSupportedMultimodel() const
{
  // If package includes no-connection model, model_count is less than real model count in package.
  // Then this method will throw exception based on model index
  //  1st model: input assumption
  //  Otherwise: edges assumption

  // Assumption: edges
  // m1 < m2, s1 == 0 and s2 == 0 if edge 'm1:s1:o1 -> m2:s2:o2'
  for (auto &&edge : _model_edges->edges)
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
      for (const auto &[model_index, subg_index, io_index] : _model_edges->pkg_inputs)
      {
        if ((model_index == ir::ModelIndex{0}) || (subg_index == ir::SubgraphIndex{0}) ||
            (io_index == input_index))
          return true;
      }

      return false;
    };

    for (uint32_t i = 0; i < first_executor->inputSize(); i++)
    {
      if (!search_first_model(ir::IOIndex{i}))
        throw std::runtime_error{"Cannot find 1st model's input buffer"};
    }
  }

  // Check whether nnpkg outputs and Edge `from` are duplicated
  for (const auto &edge : _model_edges->edges)
  {
    if (std::find(_model_edges->pkg_outputs.begin(), _model_edges->pkg_outputs.end(), edge.from) !=
        _model_edges->pkg_outputs.end())
    {
      throw std::runtime_error{"Multi model execution does not support duplicating nnpkg outputs "
                               "with `from` of edges yet"};
    }
  }
}

void MultiModelExecutors::createEdgeTensors()
{
  if (_is_created_edge_tensors)
  {
    return;
  }

  // Create EdgeTensor for edges between executors
  for (const auto &pair : _edge_map)
  {
    const auto &from_iodesc = pair.first;
    const auto &[from_model_index, from_subg_index, from_io_index] = from_iodesc;

    const auto from_executor = _executors.at({from_model_index, from_subg_index}).get();
    const auto &from_info = from_executor->outputInfo(from_io_index.value());
    _edge_tensors[from_iodesc] = std::make_unique<EdgeTensor>(from_info);
  }

  _is_created_edge_tensors = true;
}

void MultiModelExecutors::CreatePkgIOTensors(const IODescription &desc)
{
  for (const auto &pkg_input : _model_edges->pkg_inputs)
  {
    // Create IOTensor for nnpkg inputs
    const auto &[model_index, subg_index, io_index] = pkg_input;
    const auto input_pkg_index =
      find_input_index(_model_edges->pkg_inputs, model_index, subg_index, io_index);
    if (input_pkg_index == -1)
      throw std::runtime_error{"Cannot find multi model input index"};
    auto &input_desc = desc.inputs[input_pkg_index];
    // TODO Remove const_cast (we need const_cast as ITensor is writable)
    _pkg_input_tensors[pkg_input] = std::make_unique<backend::builtin::UserTensor>(
      input_desc.info, const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(input_desc.buffer)),
      input_desc.size);
  }

  for (const auto &pkg_output : _model_edges->pkg_outputs)
  {
    // Create IOTensor for nnpkg outputs
    const auto &[model_index, subg_index, io_index] = pkg_output;
    const auto output_pkg_index =
      find_output_index(_model_edges->pkg_outputs, model_index, subg_index, io_index);
    if (output_pkg_index == -1)
      throw std::runtime_error{"Cannot find multi model output index"};
    const auto output_io_tensor = dynamic_cast<const backend::builtin::IOTensor *>(
      at(model_index, subg_index)->outputTensor(io_index.value()));
    if (!output_io_tensor)
      throw std::runtime_error{"Output tensor must be IOTensor"};

    bool skip_set_output = output_io_tensor->hasBackendTensor();
    auto &output_desc = desc.outputs[output_pkg_index];
    // If buffer is nullptr, output is optional or internally allocated buffer,
    // and optional output's size is 0
    if (output_desc.buffer == nullptr &&
        (output_desc.size != 0 || output_desc.info.total_size() != 0) && !skip_set_output)
      throw std::runtime_error{"Output " + std::to_string(output_pkg_index) +
                               "'s buffer is not set."};

    _pkg_output_tensors[pkg_output] = std::make_unique<backend::builtin::UserTensor>(
      output_desc.info, reinterpret_cast<uint8_t *>(output_desc.buffer), output_desc.size);
  }
}

void MultiModelExecutors::execute(ExecutionContext &ctx)
{
  auto &desc = ctx.desc;

  // Check supported multi model package
  checkSupportedMultimodel();

  // TODO Move creating edge tensors in compilation stage
  createEdgeTensors();

  // TODO Create IOTensors only once and recreate them only if nnpkg info changes
  CreatePkgIOTensors(desc);

  // TODO Find better way to schedule order of executors
  auto const model_count = modelCount();

  auto find_from = [&](const ir::ModelIndex &model_index, const ir::SubgraphIndex &subg_index,
                       const ir::IOIndex &io_index) {
    for (const auto &edge : _model_edges->edges)
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

    // Set IOTensors
    // TODO Set internal IOTensors only once
    std::vector<backend::IPortableTensor *> inputs_inter;
    std::vector<backend::IPortableTensor *> outputs_inter;
    auto const input_size = executor->inputSize();
    auto const output_size = executor->outputSize();
    inputs_inter.resize(input_size);
    outputs_inter.resize(output_size);

    // Set inputs of executor
    // TODO Create layer to allocate/deallocate buffers of EdgeTensor for each executor
    for (uint32_t i = 0; i < input_size; i++)
    {
      const auto input_pkg_index = find_input_index(_model_edges->pkg_inputs, model_index,
                                                    ir::SubgraphIndex{0}, ir::IOIndex{i});
      const auto input_io_desc = ir::IODesc{model_index, ir::SubgraphIndex{0}, ir::IOIndex{i}};
      if (input_pkg_index != -1)
      {
        inputs_inter[i] = _pkg_input_tensors[input_io_desc].get();
      }
      else
      {
        auto from_iodesc = find_from(model_index, ir::SubgraphIndex{0}, ir::IOIndex{i});

        // Supported only sequantial execution of models
        assert(std::get<ir::ModelIndex>(from_iodesc).value() < model_index.value());
        assert(std::get<ir::SubgraphIndex>(from_iodesc).value() == 0);
        inputs_inter[i] = _edge_tensors.at(from_iodesc).get();
        assert(inputs_inter[i]->buffer() != nullptr);
      }
    }

    // Set outputs of executor
    for (uint32_t i = 0; i < output_size; i++)
    {
      const auto output_pkg_index = find_output_index(_model_edges->pkg_outputs, model_index,
                                                      ir::SubgraphIndex{0}, ir::IOIndex{i});
      const auto output_io_desc = ir::IODesc{model_index, ir::SubgraphIndex{0}, ir::IOIndex{i}};
      if (output_pkg_index != -1)
      {
        outputs_inter[i] = _pkg_output_tensors[output_io_desc].get();
      }
      else
      {
        // Allocate buffer of `from` tensors
        const auto from_iodesc = ir::IODesc{model_index, ir::SubgraphIndex{0}, ir::IOIndex{i}};
        _edge_tensors[from_iodesc]->allocate_buffer();
        outputs_inter[i] = _edge_tensors[from_iodesc].get();

        // Increase reference count of `from` tensors for edges
        for (uint32_t j = 0; j < _edge_map[from_iodesc].size(); j++)
          _edge_tensors[from_iodesc]->increase_ref();
      }
    }

    executor->execute(inputs_inter, outputs_inter, ctx.options);

    // Release input buffers that are no longer needed
    for (uint32_t i = 0; i < input_size; i++)
    {
      const auto input_pkg_index = find_input_index(_model_edges->pkg_inputs, model_index,
                                                    ir::SubgraphIndex{0}, ir::IOIndex{i});
      if (input_pkg_index == -1)
      {
        // Decrease reference count of `from` tensor if input tensor is the `from` tensor
        const auto from_iodesc = find_from(model_index, ir::SubgraphIndex{0}, ir::IOIndex{i});
        _edge_tensors[from_iodesc]->decrease_ref();
      }
    }

    // Get dynamic shape inference result
    for (uint32_t i = 0; i < output_size; i++)
    {
      const auto output_pkg_index = find_output_index(_model_edges->pkg_outputs, model_index,
                                                      ir::SubgraphIndex{0}, ir::IOIndex{i});

      if (output_pkg_index != -1)
      {
        const auto output_io_tensor =
          dynamic_cast<const backend::builtin::IOTensor *>(outputTensor(ir::IOIndex{i}));
        if (!output_io_tensor)
          throw std::runtime_error{"Output tensor must be IOTensor"};

        ctx.desc.outputs[output_pkg_index].info.shape(output_io_tensor->get_info().shape());
      }
    }
  }
}

// modelCount() iterates _executors.
// It assumes that Compiler will generate Executor for all models and _executors includes all
// generated Executor.
// If nnpackage includes model(s) which has no connection and Compiler does not
// generate Executor for them, modelCount() return less value than real model count.
uint16_t MultiModelExecutors::modelCount() const
{
  uint16_t model_count = 0;
  for (; _executors.find(std::make_pair(ir::ModelIndex{model_count}, ir::SubgraphIndex{0})) !=
         _executors.end();
       model_count++)
    ;

  return model_count;
}

} // namespace onert::exec
