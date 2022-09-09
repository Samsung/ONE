/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "loader/GraphLoader.h"

#include "loader/KernelBuilder.h"

namespace luci_interpreter
{

GraphLoader::GraphLoader(CircleReader *reader, RuntimeGraph *runtime_graph,
                         IMemoryManager *memory_manager,
                         std::unordered_map<int32_t, Tensor *> *index_to_tensor)
  : _reader(reader), _runtime_graph(runtime_graph), _memory_manager(memory_manager),
    _index_to_tensor(index_to_tensor)
{
}

// TODO: handle with static manager
void GraphLoader::loadTensors()
{
  for (uint32_t i = 0; i < _reader->tensors().size(); ++i)
  {
    const auto const_tensor = _reader->tensors().at(i);

    // TODO: handle with variable tensors
    if (const_tensor->is_variable())
      continue;

    auto const buffer = wrap(_reader->buffers()[const_tensor->buffer()]->data());
    auto const const_dims = wrap(const_tensor->shape()); // in NHWC
    if (const_dims.empty() && buffer.empty())
    {
      // unknown shape tensor and scalar tensor
      continue;
    }

    uint32_t size = 1;
    for (int const_dim : const_dims)
    {
      size *= const_dim;
    }

    if (buffer.empty() && size > 0)
    {
      // normal empty tensor
      continue;
    }

    Shape shape(static_cast<int>(const_dims.size()));
    for (int j = 0; j < const_dims.size(); ++j)
    {
      shape.dim(j) = const_dims.at(j);
    }

    //  Create dtype
    const auto dtype = luci_datatype(const_tensor->type());

    AffineQuantization quantization;
    if (dtype == DataType::U8 or dtype == DataType::S8 or dtype == DataType::S16)
    {
      const auto quant_params = const_tensor->quantization();
      assert(quant_params->zero_point()->size() == quant_params->scale()->size());
      quantization.scale.assign(quant_params->scale()->cbegin(), quant_params->scale()->cend());
      quantization.zero_point.assign(quant_params->zero_point()->cbegin(),
                                     quant_params->zero_point()->cend());
      quantization.quantized_dimension = quant_params->quantized_dimension();
    }

    // Get pointer to data from buffer
    auto data_ptr = const_cast<unsigned char *>(buffer.data());

    size *= getDataTypeSize(dtype);

    auto tensor = std::make_unique<Tensor>(dtype, std::move(shape), std::move(quantization),
                                           const_tensor->name()->str());
    // Save pointer to const data
    tensor->writeDataWithoutCopy(static_cast<void *>(data_ptr));

    _index_to_tensor->emplace(i, tensor.get());
    _runtime_graph->addTensor(std::move(tensor));
  }
}

void GraphLoader::initInputTensors() const
{
  for (const auto input_ind : _reader->inputs())
  {
    const auto tensor = _reader->tensors()[input_ind];
    const auto dtype = luci_datatype(tensor->type());
    const auto tensor_shape = wrap(tensor->shape());

    Shape shape(static_cast<int>(tensor_shape.size()));
    for (int i = 0; i < tensor_shape.size(); ++i)
    {
      shape.dim(i) = tensor_shape.at(i);
    }

    AffineQuantization quantization;
    if (dtype == DataType::U8 or dtype == DataType::S8 or dtype == DataType::S16)
    {
      const auto quant_params = tensor->quantization();
      assert(quant_params->zero_point()->size() == quant_params->scale()->size());
      quantization.scale.assign(quant_params->scale()->cbegin(), quant_params->scale()->cend());
      quantization.zero_point.assign(quant_params->zero_point()->cbegin(),
                                     quant_params->zero_point()->cend());
      quantization.quantized_dimension = quant_params->quantized_dimension();
    }
    auto tensor_interpreter = std::make_unique<Tensor>(
      dtype, std::move(shape), std::move(quantization), tensor->name()->str());
    _memory_manager->allocate_memory(*tensor_interpreter);

    _runtime_graph->addInputTensor(tensor_interpreter.get());
    _index_to_tensor->emplace(input_ind, tensor_interpreter.get());
    _runtime_graph->addTensor(std::move(tensor_interpreter));
  }
}

void GraphLoader::loadOperators()
{
  KernelBuilder kernel_builder(_runtime_graph, _reader);

  for (uint32_t i = 0; i < _reader->operators().size(); ++i)
  {
    const auto op = _reader->operators().at(i);
    assert(op != nullptr);

    std::vector<std::pair<const Tensor *, int32_t>> input_tensors;
    std::vector<std::pair<Tensor *, int32_t>> output_tensors;

    for (int32_t j = 0; j < op->inputs()->size(); ++j)
    {
      const auto input_index = op->inputs()->operator[](j);
      if (_index_to_tensor->find(input_index) != _index_to_tensor->end())
      {
        auto input_tensor = const_cast<const Tensor *>(_index_to_tensor->at(input_index));
        input_tensors.emplace_back(input_tensor, input_index);
      }
      else
      {
        input_tensors.emplace_back(nullptr, input_index);
      }
    }

    for (int32_t j = 0; j < op->outputs()->size(); ++j)
    {
      const auto output_index = op->outputs()->operator[](j);

      const auto tensor = _reader->tensors()[output_index];
      const auto dtype = luci_datatype(tensor->type());
      const auto tensor_shape = wrap(tensor->shape());

      Shape shape(static_cast<int>(tensor_shape.size()));
      for (int k = 0; k < tensor_shape.size(); ++k)
      {
        shape.dim(k) = tensor_shape.at(k);
      }

      AffineQuantization quantization;
      if (dtype == DataType::U8 or dtype == DataType::S8 or dtype == DataType::S16)
      {
        const auto quant_params = tensor->quantization();
        assert(quant_params->zero_point()->size() == quant_params->scale()->size());
        quantization.scale.assign(quant_params->scale()->cbegin(), quant_params->scale()->cend());
        quantization.zero_point.assign(quant_params->zero_point()->cbegin(),
                                       quant_params->zero_point()->cend());
        quantization.quantized_dimension = quant_params->quantized_dimension();
      }
      auto tensor_interpreter = std::make_unique<Tensor>(
        dtype, std::move(shape), std::move(quantization), tensor->name()->str());

      _index_to_tensor->emplace(output_index, tensor_interpreter.get());
      output_tensors.emplace_back(tensor_interpreter.get(), output_index);

      if (std::find(_reader->outputs().begin(), _reader->outputs().end(), output_index) !=
          _reader->outputs().end())
        _runtime_graph->addOutputTensor(tensor_interpreter.get());

      _runtime_graph->addTensor(std::move(tensor_interpreter));
    }

    std::unique_ptr<Kernel> kernel = kernel_builder.build(input_tensors, output_tensors, i);
    _runtime_graph->addKernel(std::move(kernel));
  }
}

} // namespace luci_interpreter
