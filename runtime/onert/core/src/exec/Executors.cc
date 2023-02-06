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

class Executors::EdgeTensor : public backend::builtin::IOTensor
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

uint32_t Executors::inputSize() const { return _model_edges->pkg_inputs.size(); }

uint32_t Executors::outputSize() const { return _model_edges->pkg_outputs.size(); }

const ir::OperandInfo &Executors::inputInfo(const ir::IOIndex &index) const
{
  auto const desc = _model_edges->pkg_inputs[index.value()];
  auto const model_index = std::get<0>(desc);
  auto const subg_index = std::get<1>(desc);
  auto const io_index = std::get<2>(desc);
  auto const executor = at(model_index, subg_index);
  return executor->getInputTensors().at(io_index.value())->orig_info();
}

const ir::OperandInfo &Executors::outputInfo(const ir::IOIndex &index) const
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

    for (uint32_t i = 0; i < first_executor->getInputTensors().size(); i++)
    {
      if (!search_first_model(ir::IOIndex{i}))
        throw std::runtime_error{"Cannot find 1st model's input buffer"};
    }
  }
}

// TODO Support type-aware quantization of nnpkg inputs/outputs
void Executors::createTypeAwareQuantLayers()
{
  if (_is_created_type_quant_layers)
  {
    return;
  }

  // Initialize layers for type-ware quantization as empty layer
  for (const auto &pair : _executors)
  {
    const auto &model_index = pair.first.first;
    const auto &subg_index = pair.first.second;

    std::vector<backend::ITensor *> inputs;
    std::vector<backend::ITensor *> outputs;
    _type_aware_quant_layers[{model_index, subg_index}] =
      std::make_unique<PermuteLayer>(inputs, outputs);
  }

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

    const auto &to_list = pair.second;
    std::vector<backend::ITensor *> inputs;
    std::vector<backend::ITensor *> outputs;
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
        const auto &to_info = to_executor->getInputTensors().at(to_io_index.value())->orig_info();
        const auto to_layout = to_tensor->orig_layout();
        inputs.emplace_back(from_tensor);

        auto type_aware_quant_tensor = std::make_unique<EdgeTensor>(to_info, to_layout);
        outputs.emplace_back(type_aware_quant_tensor.get());

        _type_aware_quant_tensors[to_iodesc] = std::move(type_aware_quant_tensor);
      }
    }

    auto layer = std::make_unique<PermuteLayer>(inputs, outputs);
    layer->prepare();
    _type_aware_quant_layers[{from_model_index, from_subg_index}] = std::move(layer);
  }

  _is_created_type_quant_layers = true;
}

void Executors::execute(const IODescription &desc)
{
  // Check supported multi model package
  checkSupportedMultimodel();

  // TODO Move creating layers for type-aware quantization in compilation stage
  createTypeAwareQuantLayers();

  // TODO Find better way to schedule order of executors
  std::vector<std::unique_ptr<backend::builtin::IOTensor>> pkgs_inputs(desc.inputs.size());
  std::vector<std::unique_ptr<backend::builtin::IOTensor>> pkgs_outputs(desc.outputs.size());
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

    // Set IOTensors
    std::vector<backend::IPortableTensor *> inputs_inter;
    std::vector<backend::IPortableTensor *> outputs_inter;
    const auto &input_tensors = executor->getInputTensors();
    const auto &output_tensors = executor->getOutputTensors();
    auto const input_size = input_tensors.size();
    auto const output_size = output_tensors.size();
    inputs_inter.resize(input_size);
    outputs_inter.resize(output_size);

    // Set inputs of executor
    for (uint32_t i = 0; i < input_size; i++)
    {
      auto const &info = input_tensors.at(i)->orig_info();

      auto input_pkg_index = find_input_index(model_index, ir::SubgraphIndex{0}, ir::IOIndex{i});
      if (input_pkg_index != -1)
      {
        auto input_desc = desc.inputs[input_pkg_index].get();
        const auto orig_layout = input_tensors.at(i)->orig_layout();
        if (input_desc->layout != orig_layout)
        {
          throw std::runtime_error("Executors.cc: Changing layout is not supported");
        }
        pkgs_inputs[input_pkg_index] =
          std::make_unique<backend::builtin::IOTensor>(info, input_desc->layout);
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
        const auto from_executor = _executors.at({from_model_index, from_subg_index}).get();
        const auto to_iodesc = ir::IODesc{model_index, ir::SubgraphIndex{0}, ir::IOIndex{i}};
        if (_type_aware_quant_tensors.find(to_iodesc) == _type_aware_quant_tensors.end())
        {
          inputs_inter[i] = from_executor->getOutputTensors().at(from_ioindex);
        }
        else
        {
          inputs_inter[i] = _type_aware_quant_tensors.at(to_iodesc).get();
        }
        assert(inputs_inter[i]->buffer() != nullptr);
      }
    }

    // Set outputs of executor
    for (uint32_t i = 0; i < output_size; i++)
    {
      const auto &output_tensor = output_tensors.at(i);
      auto const &info = output_tensor->orig_info();

      auto output_pkg_index = find_output_index(model_index, ir::SubgraphIndex{0}, ir::IOIndex{i});
      if (output_pkg_index != -1)
      {
        auto output_desc = desc.outputs[output_pkg_index].get();
        const auto orig_layout = output_tensors.at(i)->orig_layout();
        if (output_desc->layout != orig_layout)
        {
          throw std::runtime_error("Executors.cc: Changing layout is not supported");
        }
        pkgs_outputs[output_pkg_index] =
          std::make_unique<backend::builtin::IOTensor>(info, output_desc->layout);
        pkgs_outputs[output_pkg_index]->setUserTensor(
          reinterpret_cast<uint8_t *>(output_desc->buffer), output_desc->size);

        outputs_inter[i] = pkgs_outputs[output_pkg_index].get();
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
          if (_type_aware_quant_tensors.find(to_iodesc) != _type_aware_quant_tensors.end())
          {
            auto type_aware_quant_tensor = _type_aware_quant_tensors.at(to_iodesc).get();
            type_aware_quant_tensor->allocate_buffer();

            _edge_tensors[from_iodesc]->decrease_ref();
          }
        }
      }
    }

    executor->execute(inputs_inter, outputs_inter);
    _type_aware_quant_layers[{model_index, ir::SubgraphIndex{0}}]->run();

    // Release input buffers that are no longer needed
    for (uint32_t i = 0; i < input_size; i++)
    {
      const auto input_pkg_index =
        find_input_index(model_index, ir::SubgraphIndex{0}, ir::IOIndex{i});

      const auto to_iodesc = ir::IODesc{model_index, ir::SubgraphIndex{0}, ir::IOIndex{i}};
      if (input_pkg_index == -1)
      {
        if (_type_aware_quant_tensors.find(to_iodesc) != _type_aware_quant_tensors.end())
        {
          // Decrease reference count of tensor for type-aware quantization if input tensor is the
          // tensor
          const auto to_iodesc = ir::IODesc{model_index, ir::SubgraphIndex{0}, ir::IOIndex{i}};
          if (_type_aware_quant_tensors.find(to_iodesc) != _type_aware_quant_tensors.end())
          {
            _type_aware_quant_tensors[to_iodesc]->decrease_ref();
          }
        }
        else
        {
          // Decrease reference count of `from` tensor if input tensor is the `from` tensor
          const auto from_iodesc = find_from(model_index, ir::SubgraphIndex{0}, ir::IOIndex{i});
          _edge_tensors[from_iodesc]->decrease_ref();
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
          return _type_aware_quant_tensors.find(to_iodesc) == _type_aware_quant_tensors.end();
        });

      if (to_be_release)
      {
        // This edge tensor's buffer won't be used in other executors
        // Tensors for type-aware quantization take over the role of this edge tensor instead
        _edge_tensors[from_iodesc]->decrease_ref();
      }
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
