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

#include "backend/ITensor.h"
#include "util/logging.h"

namespace onert
{
namespace exec
{

ExecutorBase::ExecutorBase(std::unique_ptr<ir::LoweredGraph> &&lowered_graph,
                           const compiler::TensorBuilders &tensor_builders)
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

      // Controlflow opeartion can make subgraph has unused input.
      assert(tensor != nullptr || _lowered_graph->graph().getInputs().contains(ind));
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
        {
          if (tensor_builder->supportDynamicTensor())
          {
            DynAllocInfo dyn_alloc_info{ind, tensor_builder->dynamicTensorManager()};
            _output_to_dyn_alloc_info.emplace(tensor, dyn_alloc_info);
          }
          break;
        }
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
    case DataType::QUANT_UINT8_ASYMM:
    case DataType::UINT8:
      return source<uint8_t>(index, buffer, length, io_layout);
    case DataType::QUANT_INT8_SYMM:
      return source<int8_t>(index, buffer, length, io_layout);
    case DataType::INT64:
      return source<int64_t>(index, buffer, length, io_layout);
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
    case DataType::QUANT_UINT8_ASYMM:
    case DataType::UINT8:
      return sink<uint8_t>(index, buffer, length, io_layout);
    case DataType::QUANT_INT8_SYMM:
      return sink<int8_t>(index, buffer, length, io_layout);
    case DataType::INT64:
      return sink<int64_t>(index, buffer, length, io_layout);
    default:
      throw std::runtime_error("Not supported yet");
  }
}

void ExecutorBase::execute(const std::vector<std::shared_ptr<backend::ITensor>> &src_tensors,
                           const std::shared_ptr<IPermuteFunction> &pre_fn)
{
  // For thread-safe, use mutex
  // TODO: if all used backends on this executor are thread-safe,
  //       do not need to use mutex (otherwise, use mutex)
  // Deadlock occurs when an Executor is called recursively.
  std::lock_guard<std::mutex> lock(_mutex);

  assert(src_tensors.size() == _graph.getInputs().size());
  assert(src_tensors.size() == _input_tensors.size());
  for (uint32_t n = 0; n < _graph.getInputs().size(); ++n)
  {
    // when user changes input shape, the input tensor is dynamic and its memory is not allocated.
    // This code find the info to allocate dynamic tensor, and allocate memory based on the source
    // tensor's shape set by caller.
    const auto src_tensor = src_tensors[n];
    auto input_tensor = _input_tensors[n];
    // If src_tensor or input_tensor is nullptr, pre_fn does not copy the tensors
    if (src_tensor != nullptr && input_tensor != nullptr)
    {
      auto dyn_alloc_info = _input_to_dyn_alloc_info.find(_input_tensors[n]);
      const auto orig_input_shape = input_tensor->getShape();
      const auto changed_input_shape =
          convertShape(src_tensor->getShape(), src_tensor->layout(), input_tensor->layout());
      if (orig_input_shape != changed_input_shape)
      {
        if (dyn_alloc_info == _input_to_dyn_alloc_info.end())
        {
          // The input_tensor is a dynamic tensor of backend that doesn't support dynamic tensor
          throw std::runtime_error("Unknown dim is found at execution time for a backend that "
                                   "does not support dynamic tensor");
        }
        else
        {
          input_tensor->set_dynamic();
        }
      }
    }
  }

  // TODO Move calling permute_fn.run() into executeImpl()
  assert(pre_fn);
  pre_fn->run();

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

    // If nnfw_set_input_tensorinfo() was called for an input, set change and prepare memory
    handleDynamicInputTensor(input_index, desc);

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
    auto &output = *desc.outputs.at(n);

    // set shape of outputDesc to tensor shape since tensor can be dynamic
    const auto output_tensor_shape = _output_tensors[n]->getShape();
    output.info.shape(
        convertShape(output_tensor_shape, _output_tensors[n]->layout(), output.layout));

    size_t sink_length =
        output.info.shape().num_elements() * ir::sizeOfDataType(output.info.typeInfo().type());
    assert(sink_length ==
           _output_tensors[n]->getShape().num_elements() *
               ir::sizeOfDataType(_output_tensors[n]->data_type()));
    if (output.size < sink_length)
      throw std::runtime_error("ExecutorBase: output buffer size is less than output tensor size");

    sinks.at(n) =
        sink(output_index, output.info.typeInfo(), output.buffer, sink_length, output.layout);

    auto getter = [&](::onert::backend::ITensor &tensor) { sinks.at(n)->pull(tensor); };

    _output_tensors[n]->access(getter);

    // deallocate output tensors if it is dynamic
    {
      auto find = _output_to_dyn_alloc_info.find(_output_tensors[n]);
      if (find != _output_to_dyn_alloc_info.end())
      {
        auto &dyn_alloc_info = find->second;
        auto *dyn_tensor_mgr = dyn_alloc_info.dyn_tensor_manager;
        auto outut_ind = dyn_alloc_info.ind;

        dyn_tensor_mgr->deallocSubgraphOutput(outut_ind);
      }
    }
  }
}

/**
 * @brief Changes tensor shape and allocate memory
 *        if input shape was changed by nnfw_set_input_tensorinfo()
 *
 * @note  Cases are:
 *        1) static operand -> nnfw_set_input_tensorinfo() -> execute() -> execute()
 *                                                        (a)          (b)
 *
 *           at (a), operand is static, tensor is static - memory dealloc is not needed
 *                   (DynamicTensorManager cannot dealloc memory allocated by StaticTensorManager)
 *           at (b), operand is static, tensor is dynamic - memory dealloc is needed
 *
 *        2) dynamic operand -> nnfw_set_input_tensorinfo() -> execute() -> execute()
 *                                                         (a)          (b)
 *
 *           at (a), operand is dynamic, tensor is dynamic - memory dealloc is not needed
 *                                                           since it has not been allocated yet
 *           at (b), operand is dynamic, tensor is dynamic - memory dealloc is needed
 */
void ExecutorBase::handleDynamicInputTensor(ir::IOIndex io_ind, const IODescription &desc)
{
  auto shape_sig_found = desc.input_shape_signature.find(io_ind);
  if (shape_sig_found != desc.input_shape_signature.end())
  {
    auto dyn_alloc_info = _input_to_dyn_alloc_info.find(_input_tensors[io_ind.value()]);
    if (dyn_alloc_info == _input_to_dyn_alloc_info.end())
      throw std::runtime_error("Unknown dim is found at execution time for a backend that "
                               "does not support dynamic tensor");

    auto changed_input_shape = shape_sig_found->second;
    auto operand_ind = dyn_alloc_info->second.ind;

    dyn_alloc_info->second.dyn_tensor_manager->applyShape(operand_ind, changed_input_shape);
  }
}

} // namespace exec
} // namespace onert
