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
    auto &input_desc = pkg_inputs[i];
    if ((std::get<ir::ModelIndex>(input_desc) == model_index) &&
        (std::get<ir::SubgraphIndex>(input_desc) == subg_index) &&
        (std::get<ir::IOIndex>(input_desc) == io_index))
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
    auto &input_desc = pkg_outputs[i];
    if ((std::get<ir::ModelIndex>(input_desc) == model_index) &&
        (std::get<ir::SubgraphIndex>(input_desc) == subg_index) &&
        (std::get<ir::IOIndex>(input_desc) == io_index))
      return static_cast<int32_t>(i);
  }
  return -1;
}

} // namespace

namespace onert
{
namespace exec
{

class MultiModelExecutors::EdgeTensor : public backend::builtin::IOTensor
{
public:
  EdgeTensor(const ir::OperandInfo &info, ir::Layout layout)
    : backend::builtin::IOTensor(info, layout), _buffer{nullptr}, _ref_count{0}
  {
  }
  ~EdgeTensor() = default;

  void allocate_buffer()
  {
    const auto total_size = orig_info().total_size();
    _buffer = std::make_unique<uint8_t[]>(total_size);
    _ref_count = 1;

    // NOTE Executor's inputs/outputs are always IPortableTensor. If backend of inputs/outputs
    //      is using tensor that does not inherit IPortableTensor, Permute operation is added
    //      and all inputs/outputs become IPortableTensor at compile stage.
    //      This allows user's buffers to be set to inputs/outputs of executors.
    setUserTensor(_buffer.get(), total_size);
  }

  void increase_ref() { _ref_count++; }

  void decrease_ref()
  {
    assert(_ref_count > 0);
    _ref_count--;
    if (_ref_count == 0)
    {
      _buffer.reset();
      setUserTensor(nullptr, orig_info().total_size());
    }
  }

private:
  std::unique_ptr<uint8_t[]> _buffer;
  int32_t _ref_count;
};

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
  auto const desc = _model_edges->pkg_inputs[index.value()];
  auto const model_index = std::get<0>(desc);
  auto const subg_index = std::get<1>(desc);
  auto const io_index = std::get<2>(desc);
  auto const executor = at(model_index, subg_index);
  return executor->getInputTensors().at(io_index.value())->orig_info();
}

const ir::OperandInfo &MultiModelExecutors::outputInfo(const ir::IOIndex &index) const
{
  auto const desc = _model_edges->pkg_outputs[index.value()];
  auto const model_index = std::get<0>(desc);
  auto const subg_index = std::get<1>(desc);
  auto const io_index = std::get<2>(desc);
  auto const executor = at(model_index, subg_index);
  return executor->getOutputTensors().at(io_index.value())->orig_info();
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
      for (const auto &input : _model_edges->pkg_inputs)
      {
        if ((std::get<ir::ModelIndex>(input) == ir::ModelIndex{0}) ||
            (std::get<ir::SubgraphIndex>(input) == ir::SubgraphIndex{0}) ||
            (std::get<ir::IOIndex>(input) == input_index))
          return true;
      }

      return false;
    };

    for (uint32_t i = 0; i < first_executor->getInputTensors().size(); i++)
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

void MultiModelExecutors::createEdgeQuantLayers()
{
  if (_is_created_edge_quant_layers)
  {
    return;
  }

  // Create EdgeTensor for edges between executors
  for (const auto &pair : _edge_map)
  {
    const auto &from_iodesc = pair.first;
    const auto &from_model_index = std::get<ir::ModelIndex>(from_iodesc);
    const auto &from_subg_index = std::get<ir::SubgraphIndex>(from_iodesc);
    const auto &from_io_index = std::get<ir::IOIndex>(from_iodesc);

    const auto from_executor = _executors.at({from_model_index, from_subg_index}).get();
    const auto from_tensor = from_executor->getOutputTensors().at(from_io_index.value());

    const auto &from_info = from_tensor->orig_info();
    const auto from_layout = from_tensor->orig_layout();
    _edge_tensors[from_iodesc] = std::make_unique<EdgeTensor>(from_info, from_layout);
  }

  // Append type-aware quantization layer for edges between executors
  for (const auto &executor_pair : _executors)
  {
    const auto &executor_index = executor_pair.first;
    const auto &model_index = executor_index.first;
    const auto &subg_index = executor_index.second;

    std::vector<backend::ITensor *> inputs;
    std::vector<backend::ITensor *> outputs;
    for (const auto &pair : _edge_map)
    {
      const auto &from_iodesc = pair.first;
      if (std::get<ir::ModelIndex>(from_iodesc) == model_index &&
          std::get<ir::SubgraphIndex>(from_iodesc) == subg_index)
      {
        const auto from_tensor = _edge_tensors[from_iodesc].get();
        const auto &to_list = pair.second;

        for (const auto &to_iodesc : to_list)
        {
          const auto &to_model_index = std::get<ir::ModelIndex>(to_iodesc);
          const auto &to_subg_index = std::get<ir::SubgraphIndex>(to_iodesc);
          const auto &to_io_index = std::get<ir::IOIndex>(to_iodesc);

          const auto to_executor = _executors.at({to_model_index, to_subg_index}).get();
          const auto to_tensor = to_executor->getInputTensors().at(to_io_index.value());

          // TODO Unify tensors with the same `from` tensor and same type
          if (from_tensor->data_type() != to_tensor->data_type())
          {
            assert(inputs.size() == outputs.size());
            const auto &to_info =
              to_executor->getInputTensors().at(to_io_index.value())->orig_info();
            const auto to_layout = to_tensor->orig_layout();
            inputs.emplace_back(from_tensor);

            auto type_aware_quant_tensor = std::make_unique<EdgeTensor>(to_info, to_layout);
            outputs.emplace_back(type_aware_quant_tensor.get());

            _edge_quant_tensors[to_iodesc] = std::move(type_aware_quant_tensor);
          }
        }
      }
    }

    auto layer = std::make_unique<PermuteLayer>(inputs, outputs);
    layer->prepare();
    _edge_quant_layers[{model_index, subg_index}] = std::move(layer);
  }

  _is_created_edge_quant_layers = true;
}

void MultiModelExecutors::CreatePkgIOTensors(const IODescription &desc)
{
  for (const auto &pkg_input : _model_edges->pkg_inputs)
  {
    // Create IOTensor for nnpkg inputs
    const auto &model_index = std::get<ir::ModelIndex>(pkg_input);
    const auto &subg_index = std::get<ir::SubgraphIndex>(pkg_input);
    const auto &io_index = std::get<ir::IOIndex>(pkg_input);
    const auto input_pkg_index =
      find_input_index(_model_edges->pkg_inputs, model_index, subg_index, io_index);
    if (input_pkg_index == -1)
      throw std::runtime_error{"Cannot find multi model input index"};
    auto input_desc = desc.inputs[input_pkg_index].get();
    _pkg_input_tensors[pkg_input] =
      std::make_unique<backend::builtin::IOTensor>(input_desc->info, input_desc->layout);
  }

  for (const auto &pkg_output : _model_edges->pkg_outputs)
  {
    // Create IOTensor for nnpkg outputs
    const auto &model_index = std::get<ir::ModelIndex>(pkg_output);
    const auto &subg_index = std::get<ir::SubgraphIndex>(pkg_output);
    const auto &io_index = std::get<ir::IOIndex>(pkg_output);
    const auto output_pkg_index =
      find_output_index(_model_edges->pkg_outputs, model_index, subg_index, io_index);
    if (output_pkg_index == -1)
      throw std::runtime_error{"Cannot find multi model output index"};
    auto output_desc = desc.outputs[output_pkg_index].get();
    _pkg_output_tensors[pkg_output] =
      std::make_unique<backend::builtin::IOTensor>(output_desc->info, output_desc->layout);
  }
}

void MultiModelExecutors::createPkgIOQuantLayers(const IODescription &desc)
{
  // Append type-aware quantization layer for nnpkg inputs/outputs between executors
  for (const auto &pair : _executors)
  {
    const auto &executor_index = pair.first;
    const auto &model_index = executor_index.first;
    const auto &subg_index = executor_index.second;
    const auto executor = pair.second.get();

    // Find pkg inputs of current executor
    std::vector<ir::IODesc> pkg_inputs;
    for (const auto &pkg_input : _model_edges->pkg_inputs)
    {
      if (std::get<ir::ModelIndex>(pkg_input) == model_index &&
          std::get<ir::SubgraphIndex>(pkg_input) == subg_index)
      {
        pkg_inputs.emplace_back(pkg_input);
      }
    }
    std::vector<backend::ITensor *> src_tensors;
    std::vector<backend::ITensor *> dst_tensors;
    for (const auto &pkg_input : pkg_inputs)
    {
      const auto &io_index = std::get<ir::IOIndex>(pkg_input);
      const auto input_pkg_index =
        find_input_index(_model_edges->pkg_inputs, model_index, subg_index, io_index);
      if (input_pkg_index == -1)
        throw std::runtime_error{"Cannot find multi model input index"};
      auto input_desc = desc.inputs[input_pkg_index].get();

      // Create EdgeTensor for nnpkg input if type is different
      const auto input_tensor =
        executor->getInputTensors().at(std::get<ir::IOIndex>(pkg_input).value());
      const auto &orig_info = input_tensor->orig_info();
      if (input_desc->info.typeInfo().type() != input_tensor->orig_info().typeInfo().type())
      {
        const auto orig_layout = input_tensor->orig_layout();
        auto pkg_input_edge_tensor = std::make_unique<EdgeTensor>(orig_info, orig_layout);
        _pkg_input_quant_tensors[pkg_input] = std::move(pkg_input_edge_tensor);

        // Append type-aware quantization layer's inputs/outputs
        src_tensors.emplace_back(_pkg_input_tensors[pkg_input].get());
        dst_tensors.emplace_back(_pkg_input_quant_tensors[pkg_input].get());
      }
    }

    // Create type-aware quantization layer for nnpkg inputs
    auto pkg_input_layer = std::make_unique<PermuteLayer>(src_tensors, dst_tensors);
    pkg_input_layer->prepare();
    _pkg_input_quant_layers[{model_index, subg_index}] = std::move(pkg_input_layer);

    // Find pkg outputs of current executor
    std::vector<ir::IODesc> pkg_outputs;
    for (const auto &pkg_output : _model_edges->pkg_outputs)
    {
      if (std::get<ir::ModelIndex>(pkg_output) == model_index &&
          std::get<ir::SubgraphIndex>(pkg_output) == subg_index)
      {
        pkg_outputs.emplace_back(pkg_output);
      }
    }
    src_tensors.clear();
    dst_tensors.clear();
    // Create Tensors of nnpkg outputs for type-aware quantization
    for (const auto &pkg_output : pkg_outputs)
    {
      const auto &io_index = std::get<ir::IOIndex>(pkg_output);
      const auto output_pkg_index =
        find_output_index(_model_edges->pkg_outputs, model_index, subg_index, io_index);
      if (output_pkg_index == -1)
        throw std::runtime_error{"Cannot find multi model output index"};
      auto output_desc = desc.outputs[output_pkg_index].get();

      // Create EdgeTensor for nnpkg output if type is different
      const auto output_tensor =
        executor->getOutputTensors().at(std::get<ir::IOIndex>(pkg_output).value());
      const auto &orig_info = output_tensor->orig_info();
      if (output_desc->info.typeInfo().type() != output_tensor->orig_info().typeInfo().type())
      {
        const auto orig_layout = output_tensor->orig_layout();
        auto pkg_output_edge_tensor = std::make_unique<EdgeTensor>(orig_info, orig_layout);
        _pkg_output_quant_tensors[pkg_output] = std::move(pkg_output_edge_tensor);

        // Append type-aware quantization layer's inputs/outputs
        src_tensors.emplace_back(_pkg_output_quant_tensors[pkg_output].get());
        dst_tensors.emplace_back(_pkg_output_tensors[pkg_output].get());
      }
    }

    // Create type-aware quantization layer for nnpkg outputs
    auto pkg_output_layer = std::make_unique<PermuteLayer>(src_tensors, dst_tensors);
    pkg_output_layer->prepare();
    _pkg_output_quant_layers[{model_index, subg_index}] = std::move(pkg_output_layer);
  }
}

void MultiModelExecutors::execute(const ExecutionContext &ctx)
{
  auto &desc = ctx.desc;

  // Check supported multi model package
  checkSupportedMultimodel();

  // TODO Move creating type-aware quantization layers for edges in compilation stage
  createEdgeQuantLayers();

  // TODO Create IOTensors only once and recreate them only if nnpkg info changes
  CreatePkgIOTensors(desc);

  // TODO Create type-aware quantization layers only once and recreate them only if type changes
  createPkgIOQuantLayers(desc);

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
    const auto &input_tensors = executor->getInputTensors();
    const auto &output_tensors = executor->getOutputTensors();
    auto const input_size = input_tensors.size();
    auto const output_size = output_tensors.size();
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
        // Allocate type-aware quantization tensors for nnpkg inputs and set internal tensors
        if (_pkg_input_quant_tensors.find(input_io_desc) != _pkg_input_quant_tensors.end())
        {
          _pkg_input_quant_tensors[input_io_desc]->allocate_buffer();

          inputs_inter[i] = _pkg_input_quant_tensors[input_io_desc].get();
        }
        else
        {
          inputs_inter[i] = _pkg_input_tensors[input_io_desc].get();
        }

        // Set buffer of IOTensor
        auto input_desc = desc.inputs[input_pkg_index].get();
        // TODO Remove const_cast (we need const_cast as ITensor is writable)
        _pkg_input_tensors[input_io_desc]->setUserTensor(
          reinterpret_cast<uint8_t *>(const_cast<void *>(input_desc->buffer)), input_desc->size);
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
        const auto from_executor = _executors.at({from_model_index, from_subg_index}).get();
        const auto to_iodesc = ir::IODesc{model_index, ir::SubgraphIndex{0}, ir::IOIndex{i}};
        if (_edge_quant_tensors.find(to_iodesc) == _edge_quant_tensors.end())
        {
          inputs_inter[i] = from_executor->getOutputTensors().at(from_ioindex);
        }
        else
        {
          inputs_inter[i] = _edge_quant_tensors.at(to_iodesc).get();
        }
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
        // Allocate type-aware quantization tensors for nnpkg outputs and set internal tensors
        if (_pkg_output_quant_tensors.find(output_io_desc) != _pkg_output_quant_tensors.end())
        {
          _pkg_output_quant_tensors[output_io_desc]->allocate_buffer();

          outputs_inter[i] = _pkg_output_quant_tensors[output_io_desc].get();
        }
        else
        {
          outputs_inter[i] = _pkg_output_tensors[output_io_desc].get();
        }

        // Set buffer of IOTensor
        auto output_desc = desc.outputs[output_pkg_index].get();
        _pkg_output_tensors[output_io_desc]->setUserTensor(
          reinterpret_cast<uint8_t *>(output_desc->buffer), output_desc->size);
      }
      else
      {
        // Allocate buffer of `from` tensors
        const auto from_iodesc = ir::IODesc{model_index, ir::SubgraphIndex{0}, ir::IOIndex{i}};
        _edge_tensors[from_iodesc]->allocate_buffer();
        outputs_inter[i] = _edge_tensors[from_iodesc].get();

        // Allocate buffer of tensors for type-aware quantization
        for (const auto &to_iodesc : _edge_map[from_iodesc])
        {
          _edge_tensors[from_iodesc]->increase_ref();
          if (_edge_quant_tensors.find(to_iodesc) != _edge_quant_tensors.end())
          {
            auto type_aware_quant_tensor = _edge_quant_tensors.at(to_iodesc).get();
            type_aware_quant_tensor->allocate_buffer();

            _edge_tensors[from_iodesc]->decrease_ref();
          }
        }
      }
    }

    _pkg_input_quant_layers[{model_index, ir::SubgraphIndex{0}}]->run();

    executor->execute(inputs_inter, outputs_inter);

    _edge_quant_layers[{model_index, ir::SubgraphIndex{0}}]->run();
    _pkg_output_quant_layers[{model_index, ir::SubgraphIndex{0}}]->run();

    // Release input buffers that are no longer needed
    for (uint32_t i = 0; i < input_size; i++)
    {
      const auto input_pkg_index = find_input_index(_model_edges->pkg_inputs, model_index,
                                                    ir::SubgraphIndex{0}, ir::IOIndex{i});

      const auto to_iodesc = ir::IODesc{model_index, ir::SubgraphIndex{0}, ir::IOIndex{i}};
      if (input_pkg_index == -1)
      {
        if (_edge_quant_tensors.find(to_iodesc) != _edge_quant_tensors.end())
        {
          // Decrease reference count of tensor for type-aware quantization if input tensor is the
          // tensor
          const auto to_iodesc = ir::IODesc{model_index, ir::SubgraphIndex{0}, ir::IOIndex{i}};
          if (_edge_quant_tensors.find(to_iodesc) != _edge_quant_tensors.end())
          {
            _edge_quant_tensors[to_iodesc]->decrease_ref();
          }
        }
        else
        {
          // Decrease reference count of `from` tensor if input tensor is the `from` tensor
          const auto from_iodesc = find_from(model_index, ir::SubgraphIndex{0}, ir::IOIndex{i});
          _edge_tensors[from_iodesc]->decrease_ref();

          // Decrease reference count of nnpkg inputs
          if (_pkg_input_quant_tensors.find(to_iodesc) != _pkg_input_quant_tensors.end())
          {
            _pkg_input_quant_tensors[to_iodesc]->decrease_ref();
          }
        }
      }
    }

    // Release output buffers if those buffers are no longer used other executors because of
    // type-aware quantization
    // FIXME if tensors for type-aware quantization unified for the same `from` tensor and same type
    for (uint32_t i = 0; i < output_size; i++)
    {
      auto from_iodesc = ir::IODesc{model_index, ir::SubgraphIndex{0}, ir::IOIndex{i}};

      // Check if other executors will use the buffer of edge tensor
      const auto &to_list = _edge_map[from_iodesc];
      if (to_list.size() == 0)
      {
        // This condition means `from_iodesc` tensor is an output of nnpkg
        continue;
      }

      bool to_be_release =
        !std::any_of(to_list.begin(), to_list.end(), [&](const ir::IODesc &to_iodesc) {
          // This condition means another executor uses the buffer of edge tensor
          return _edge_quant_tensors.find(to_iodesc) == _edge_quant_tensors.end();
        });

      if (to_be_release)
      {
        // This edge tensor's buffer won't be used in other executors
        // Tensors for type-aware quantization take over the role of this edge tensor instead
        _edge_tensors[from_iodesc]->decrease_ref();
      }

      // Decrease reference count of nnpkg outputs
      if (_pkg_output_quant_tensors.find(from_iodesc) != _pkg_output_quant_tensors.end())
      {
        _pkg_output_quant_tensors[from_iodesc]->decrease_ref();
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

} // namespace exec
} // namespace onert
