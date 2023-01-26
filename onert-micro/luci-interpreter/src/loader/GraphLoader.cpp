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

#include "memory_managers/StaticMemoryManager.h"

namespace luci_interpreter
{
namespace
{

using ExecutionPlanTable = std::map<uint32_t, std::vector<uint32_t>>;

// TODO: add more operations
bool isCouldBeEmplaceOperation(circle::BuiltinOperator op)
{
  switch (op)
  {
    case circle::BuiltinOperator_LOGISTIC:
    case circle::BuiltinOperator_RESHAPE:
    case circle::BuiltinOperator_EXPAND_DIMS:
      return true;
    default:
      return false;
  }
}

} // namespace

GraphLoader::GraphLoader(CircleReader *reader, IBaseRuntimeGraph *runtime_graph,
                         IMemoryManager *memory_manager,
                         std::unordered_map<int32_t, Tensor *> *index_to_tensor)
  : _reader(reader), _runtime_graph(runtime_graph), _memory_manager(memory_manager),
    _index_to_tensor(index_to_tensor)
{
}

bool GraphLoader::isCouldBeEmplaceTensor(const int32_t tensor_index)
{
  uint32_t usage_count = 0;
  for (uint32_t i = 0; i < _reader->operators().size(); ++i)
  {
    const auto op = _reader->operators().at(i);
    assert(op != nullptr);

    for (int32_t j = 0; j < op->inputs()->size(); ++j)
    {
      const auto input_index = op->inputs()->operator[](j);
      if (input_index == tensor_index)
        usage_count++;

      if (usage_count > 1)
        return false;
    }
  }
  return true;
}

void GraphLoader::loadTensors(bool use_static_memory_manager)
{
  for (uint32_t i = 0; i < _reader->tensors().size(); ++i)
  {
    const auto const_tensor = _reader->tensors().at(i);

    // TODO: handle with variable tensors
    if (const_tensor->is_variable() and use_static_memory_manager)
      assert(false && "Not supported now");

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

    if (buffer.empty() && size > 0 && not const_tensor->is_variable())
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

    AffineQuantization *quantization = nullptr;
    const auto quant_params = const_tensor->quantization();
    if (quant_params)
    {
      auto unique_ptr_quantization = std::make_unique<AffineQuantization>();
      assert(quant_params->zero_point()->size() == quant_params->scale()->size());
      unique_ptr_quantization->scale.assign(quant_params->scale()->cbegin(),
                                            quant_params->scale()->cend());
      unique_ptr_quantization->zero_point.assign(quant_params->zero_point()->cbegin(),
                                                 quant_params->zero_point()->cend());
      unique_ptr_quantization->quantized_dimension = quant_params->quantized_dimension();

      quantization = _runtime_graph->addAffineQuantization(std::move(unique_ptr_quantization));
    }

    if (size == 0)
    {
      if (quantization != nullptr)
        _runtime_graph->addIntermediateTensorAffineQuantization(quantization);

      continue;
    }

    // Get pointer to data from buffer
    auto data_ptr = const_cast<unsigned char *>(buffer.data());

    auto tensor = std::make_unique<Tensor>(dtype, std::move(shape), quantization);
    // Save pointer to const data
    assert(data_ptr != nullptr or const_tensor->is_variable());

    if (data_ptr)
      tensor->writeDataWithoutCopy(static_cast<void *>(data_ptr));

    _index_to_tensor->emplace(i, tensor.get());
    _runtime_graph->addTensor(std::move(tensor));
  }
}

void GraphLoader::initInputTensors(bool use_static_memory_manager) const
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

    AffineQuantization *quantization = nullptr;
    const auto quant_params = tensor->quantization();
    if (quant_params)
    {
      auto unique_ptr_quantization = std::make_unique<AffineQuantization>();
      assert(quant_params->zero_point()->size() == quant_params->scale()->size());
      unique_ptr_quantization->scale.assign(quant_params->scale()->cbegin(),
                                            quant_params->scale()->cend());
      unique_ptr_quantization->zero_point.assign(quant_params->zero_point()->cbegin(),
                                                 quant_params->zero_point()->cend());
      unique_ptr_quantization->quantized_dimension = quant_params->quantized_dimension();

      quantization = _runtime_graph->addAffineQuantization(std::move(unique_ptr_quantization));
    }

    auto tensor_interpreter = std::make_unique<Tensor>(dtype, std::move(shape), quantization);

    tensor_interpreter->set_allocatable(_memory_manager->is_allocate_input());
    if (not use_static_memory_manager)
    {
      // Using Dynamic Allocations
      _memory_manager->allocate_memory(*tensor_interpreter);
    }
    else
    {
      // Using static allocations
      _memory_manager->allocate_input_buf();
    }

    _runtime_graph->addInputTensor(tensor_interpreter.get());
    _index_to_tensor->emplace(input_ind, tensor_interpreter.get());
    _runtime_graph->addTensor(std::move(tensor_interpreter));
  }
}

void GraphLoader::loadOperators(bool use_static_memory_manager)
{
  ExecutionPlanTable execution_plan;
  // Set execution plan for static memory manager
  if (use_static_memory_manager)
  {
    // Read metadata
    const auto metadata = _reader->metadata();
    for (auto meta : metadata)
    {
      assert(meta != nullptr);

      assert(meta->buffer() < _reader->buffers().size());
      assert(_reader->buffers()[meta->buffer()] != nullptr);
      const auto buffer = wrap(_reader->buffers()[meta->buffer()]->data());

      if (meta->name()->str().compare("ONE_execution_plan_table") == 0)
      {
        execution_plan = read_metadata::decode_execution_plan(buffer);
      }
    }
    if (execution_plan.empty())
      assert(false && "Static Memory Manager should be used with circle-execution-planner");
  }

  KernelBuilder kernel_builder(_runtime_graph, _reader);
  const uint32_t input_size = _runtime_graph->getInputTensors().size();
  const uint32_t output_size = _reader->outputs().size();

  if (use_static_memory_manager)
  {
    // Set offset for input tensors
    for (int32_t input_ind = 0; input_ind < input_size; ++input_ind)
    {
      auto input_tensor = _runtime_graph->getInputTensors().at(input_ind);
      input_tensor->set_offset(execution_plan.at(input_ind)[0]);

      _memory_manager->allocate_memory_for_input(*input_tensor);
    }
  }

  for (uint32_t i = 0; i < _reader->operators().size(); ++i)
  {
    const auto op = _reader->operators().at(i);
    assert(op != nullptr);

    std::vector<const Tensor *> input_tensors(op->inputs()->size());
    std::vector<Tensor *> output_tensors(op->outputs()->size());

    bool is_inplace = false;
    bool is_graph_input = false;
    for (int32_t j = 0; j < op->inputs()->size(); ++j)
    {
      const auto input_index = op->inputs()->operator[](j);
      if (_index_to_tensor->find(input_index) != _index_to_tensor->end())
      {
        auto input_tensor = _index_to_tensor->at(input_index);
        input_tensors.at(j) = input_tensor;
        const auto &graph_input_tensors = _runtime_graph->getInputTensors();

        is_graph_input = (std::find(graph_input_tensors.begin(), graph_input_tensors.end(),
                                    input_tensor) != graph_input_tensors.end()) or
                         is_graph_input;

        // TODO: handle Inplace Optimization with Static Memory Manager
        if (not use_static_memory_manager and
            isCouldBeEmplaceOperation(_reader->builtin_code(op)) and op->outputs()->size() == 1 and
            isCouldBeEmplaceTensor(input_index) and not is_graph_input)
        {
          is_inplace = true;
        }
      }
      else
      {
        input_tensors.at(j) = nullptr;
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

      AffineQuantization *quantization = nullptr;
      const auto quant_params = tensor->quantization();
      if (quant_params)
      {
        auto unique_ptr_quantization = std::make_unique<AffineQuantization>();
        assert(quant_params->zero_point()->size() == quant_params->scale()->size());
        unique_ptr_quantization->scale.assign(quant_params->scale()->cbegin(),
                                              quant_params->scale()->cend());
        unique_ptr_quantization->zero_point.assign(quant_params->zero_point()->cbegin(),
                                                   quant_params->zero_point()->cend());
        unique_ptr_quantization->quantized_dimension = quant_params->quantized_dimension();

        quantization = _runtime_graph->addAffineQuantization(std::move(unique_ptr_quantization));
      }

      auto tensor_interpreter = std::make_unique<Tensor>(dtype, std::move(shape), quantization);

      if (use_static_memory_manager and
          (std::find(_reader->outputs().begin(), _reader->outputs().end(), output_index) ==
           _reader->outputs().end()))
      {
        tensor_interpreter->set_offset(execution_plan.at(i + input_size + output_size).at(0));
      }

      _index_to_tensor->emplace(output_index, tensor_interpreter.get());
      output_tensors.at(j) = tensor_interpreter.get();

      if (std::find(_reader->outputs().begin(), _reader->outputs().end(), output_index) !=
          _reader->outputs().end())
        _runtime_graph->addOutputTensor(tensor_interpreter.get());

      _runtime_graph->addTensor(std::move(tensor_interpreter));
    }

    const auto opcode = _reader->builtin_code(op);
    std::unique_ptr<Kernel> kernel =
      kernel_builder.build(std::move(input_tensors), std::move(output_tensors), opcode, i);
    kernel->setInplaceValue(is_inplace);

    _runtime_graph->addKernel(std::move(kernel));
  }
  if (use_static_memory_manager)
  {
    for (int32_t ind = 0; ind < output_size; ++ind)
    {
      auto output_tensor = _runtime_graph->getOutputTensors().at(ind);
      output_tensor->set_offset(execution_plan.at(input_size + ind)[0]);
    }
  }
}

} // namespace luci_interpreter
