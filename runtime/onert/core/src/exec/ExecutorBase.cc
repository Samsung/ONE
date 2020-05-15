/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ExecutorBase.h"
#include "util/logging.h"

namespace onert
{
namespace exec
{

ExecutorBase::ExecutorBase(std::unique_ptr<ir::LoweredGraph> &&lowered_graph,
                           const backend::TensorBuilderSet &tensor_builders)
    : _lowered_graph{std::move(lowered_graph)}, _graph{_lowered_graph->graph()}, _mutex()
{
  auto build_input_tensor_list = [&](const onert::ir::OperandIndexSequence &ind_seq) {
    std::vector<std::shared_ptr<backend::ITensor>> list;
    for (auto ind : ind_seq)
    {
      std::shared_ptr<backend::ITensor> tensor;
      for (auto &tensor_builder : tensor_builders)
      {
        tensor = tensor_builder->tensorAt(ind);
        if (tensor != nullptr)
        {
          if (tensor_builder->supportDynamicTensor())
          {
            DynAllocInfo dyn_alloc_info{ind, tensor_builder->dynamicTensorManager()};
            _input_to_dyn_alloc_info.emplace(tensor, dyn_alloc_info);
          }
          break;
        }
      }
      assert(tensor != nullptr);
      list.push_back(tensor);
    }
    return list;
  };

  auto build_output_tensor_list = [&](const onert::ir::OperandIndexSequence &ind_seq) {
    std::vector<std::shared_ptr<backend::ITensor>> list;
    for (auto ind : ind_seq)
    {
      std::shared_ptr<backend::ITensor> tensor;
      for (auto &tensor_builder : tensor_builders)
      {
        tensor = tensor_builder->tensorAt(ind);
        if (tensor != nullptr)
          break;
      }
      assert(tensor != nullptr);
      list.push_back(tensor);
    }
    return list;
  };

  _input_tensors = build_input_tensor_list(_graph.getInputs());
  _output_tensors = build_output_tensor_list(_graph.getOutputs());

  // Prepare each TensorManager on each backend
  for (auto &tensor_builder : tensor_builders)
  {
    auto s_tensor_manager = tensor_builder->releaseStaticTensorManager();
    if (s_tensor_manager != nullptr)
      _tensor_mgrs.insert(std::move(s_tensor_manager));

    if (tensor_builder->supportDynamicTensor())
    {
      auto d_tensor_manager = tensor_builder->releaseDynamicTensorManager();
      if (d_tensor_manager != nullptr)
        _tensor_mgrs.insert(std::move(d_tensor_manager));
    }
  }
}

std::unique_ptr<ISource> ExecutorBase::source(const ir::IOIndex &index, const ir::TypeInfo &type,
                                              const void *buffer, size_t length,
                                              ir::Layout io_layout)
{
  using ir::DataType;
  switch (type.type())
  {
    case DataType::FLOAT32:
      return source<float>(index, buffer, length, io_layout);
    case DataType::INT32:
      return source<int32_t>(index, buffer, length, io_layout);
    case DataType::UINT32:
      return source<uint32_t>(index, buffer, length, io_layout);
    case DataType::BOOL8:
    case DataType::QUANT8_ASYMM:
    case DataType::UINT8:
      return source<uint8_t>(index, buffer, length, io_layout);
    case DataType::QUANT8_SYMM:
      return source<int8_t>(index, buffer, length, io_layout);
    default:
      throw std::runtime_error("Not supported yet");
  }
}

std::unique_ptr<ISink> ExecutorBase::sink(const ir::IOIndex &index, const ir::TypeInfo &type,
                                          void *buffer, size_t length, ir::Layout io_layout)
{
  using ir::DataType;
  switch (type.type())
  {
    case DataType::FLOAT32:
      return sink<float>(index, buffer, length, io_layout);
    case DataType::INT32:
      return sink<int32_t>(index, buffer, length, io_layout);
    case DataType::UINT32:
      return sink<uint32_t>(index, buffer, length, io_layout);
    case DataType::BOOL8:
    case DataType::QUANT8_ASYMM:
    case DataType::UINT8:
      return sink<uint8_t>(index, buffer, length, io_layout);
    case DataType::QUANT8_SYMM:
      return sink<int8_t>(index, buffer, length, io_layout);
    default:
      throw std::runtime_error("Not supported yet");
  }
}

void ExecutorBase::execute()
{
  // For thread-safe, use mutex
  // TODO: if all used backends on this executor are thread-safe,
  //       do not need to use mutex (otherwise, use mutex)
  // Deadlock occurs when an Executor is called recursively.
  std::lock_guard<std::mutex> lock(_mutex);

  executeImpl();
}

void ExecutorBase::execute(const IODescription &desc)
{
  // For thread-safe, use mutex
  // TODO: if all used backends on this executor are thread-safe,
  //       do not need to use mutex (otherwise, use mutex)
  std::lock_guard<std::mutex> lock(_mutex);

  std::vector<std::unique_ptr<ISource>> sources{_graph.getInputs().size()};
  std::vector<std::unique_ptr<ISink>> sinks{_graph.getOutputs().size()};

  // Set input(s)
  for (uint32_t n = 0; n < _graph.getInputs().size(); ++n)
  {
    ir::IOIndex input_index{n};
    ir::OperandIndex index{_graph.getInputs().at(input_index)};

    if (desc.inputs.at(n) == nullptr)
    {
      // Optional input
      continue;
    }

    const auto operand_li = _lowered_graph->getLowerInfo()->operand.at(index).get();
    if (operand_li->def_factors().empty())
    {
      // This input is not used (i.e. constant, EX. reshape's axis)
      continue;
    }

    //
    // TODO Allocate memory for input tensor when input tensor is dynamic
    // e.g.,
    //  auto dyn_alloc_info = _input_to_dyn_alloc_info.find(_input_tensors[n]);
    //  if (dyn_alloc_info != _input_to_dyn_alloc_info.end())
    //  {
    //    auto ind = dyn_alloc_info->second.ind;
    //    dyn_alloc_info->second.dyn_tensor_manager->allocate(ind, exec_time_shape);
    //  }

    const auto &input = *desc.inputs.at(n);
    sources.at(n) =
        source(input_index, input.info.typeInfo(), input.buffer, input.size, input.layout);

    auto setter = [&](::onert::backend::ITensor &tensor) { sources.at(n)->push(tensor); };

    _input_tensors[n]->access(setter);
  }

  executeImpl();

  // Get output(s)
  for (uint32_t n = 0; n < _graph.getOutputs().size(); ++n)
  {
    ir::IOIndex output_index{n};
    // Optional output
    if (desc.outputs.at(n) == nullptr)
    {
      continue;
    }
    const auto &output = *desc.outputs.at(n);
    sinks.at(n) =
        sink(output_index, output.info.typeInfo(), output.buffer, output.size, output.layout);

    auto getter = [&](::onert::backend::ITensor &tensor) { sinks.at(n)->pull(tensor); };

    _output_tensors[n]->access(getter);
  }
}

} // namespace exec
} // namespace onert
