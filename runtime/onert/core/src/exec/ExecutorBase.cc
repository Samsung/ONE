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
#include "backend/controlflow/UserTensor.h"
#include "backend/cpu_common/Tensor.h"
#include "util/logging.h"

namespace onert
{
namespace exec
{

ExecutorBase::ExecutorBase(std::unique_ptr<ir::LoweredGraph> &&lowered_graph,
                           const std::vector<std::shared_ptr<backend::ITensor>> &input_tensors,
                           const std::vector<std::shared_ptr<backend::ITensor>> &output_tensors,
                           const compiler::TensorBuilders &tensor_builders)
    : _lowered_graph{std::move(lowered_graph)}, _graph{_lowered_graph->graph()},
      _input_tensors{input_tensors}, _output_tensors{output_tensors}, _mutex()
{
  // TODO Fix the way of knowing whether it is primary or not
  bool primary_executor = !(_input_tensors.empty() && _output_tensors.empty());
  if (!primary_executor)
  {
    auto build_input_tensor_list = [&](const onert::ir::OperandIndexSequence &ind_seq) {
      std::vector<std::shared_ptr<backend::ITensor>> list;
      for (auto ind : ind_seq)
      {
        std::shared_ptr<backend::ITensor> tensor;
        for (auto &tensor_builder : tensor_builders)
        {
          auto tensor_registry = tensor_builder->tensorRegistry();
          assert(tensor_registry);
          tensor = tensor_registry->getNativeITensor(ind);
          if (tensor != nullptr)
          {
            DynAllocInfo dyn_alloc_info{ind};
            _input_to_dyn_alloc_info.emplace(tensor, dyn_alloc_info);
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
          auto tensor_registry = tensor_builder->tensorRegistry();
          assert(tensor_registry);
          tensor = tensor_registry->getNativeITensor(ind);
          if (tensor != nullptr)
          {
            DynAllocInfo dyn_alloc_info{ind};
            _output_to_dyn_alloc_info.emplace(tensor, dyn_alloc_info);
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
  }
  else
  {
    // If primary graph, all the inputs and outputs belong to controlflow backend
    auto cf_dyn_tensor_builder = tensor_builders.getControlflowTensorBuilder();
    assert(cf_dyn_tensor_builder);

    assert(input_tensors.size() == _graph.getInputs().size());
    assert(output_tensors.size() == _graph.getOutputs().size());
    for (uint32_t i = 0; i < input_tensors.size(); i++)
    {
      auto tensor = input_tensors[i];
      auto ind = _graph.getInputs().at(i);
      DynAllocInfo dyn_alloc_info{ind};
      _input_to_dyn_alloc_info.emplace(tensor, dyn_alloc_info);
    }
    for (uint32_t i = 0; i < output_tensors.size(); i++)
    {
      auto tensor = output_tensors[i];
      auto ind = _graph.getOutputs().at(i);
      DynAllocInfo dyn_alloc_info{ind};
      _output_to_dyn_alloc_info.emplace(tensor, dyn_alloc_info);
    }
  }

  // Prepare each TensorManager on each backend
  for (auto &tensor_builder : tensor_builders)
  {
    auto s_tensor_manager = tensor_builder->releaseStaticTensorManager();
    if (s_tensor_manager != nullptr)
      _tensor_mgrs.insert(std::move(s_tensor_manager));

    auto d_tensor_manager = tensor_builder->releaseDynamicTensorManager();
    if (d_tensor_manager != nullptr)
      _tensor_mgrs.insert(std::move(d_tensor_manager));
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

  // Set input(s)
  assert(_input_tensors.size() == desc.inputs.size());
  for (uint32_t i = 0; i < _input_tensors.size(); ++i)
  {
    // TODO Remove dynamic_cast
    auto tensor = std::dynamic_pointer_cast<backend::controlflow::UserTensor>(_input_tensors[i]);
    assert(tensor);
    auto input_shape = desc.input_shape_signature.find(ir::IOIndex{i});
    if (input_shape != desc.input_shape_signature.end())
    {
      tensor->set_dynamic();
      tensor->setShape(input_shape->second);
    }
    // TODO Better design for ITensor? (we need const_cast as ITensor is writable)
    tensor->setBuffer(static_cast<uint8_t *>(const_cast<void *>(desc.inputs[i]->buffer)),
                      desc.inputs[i]->size);

    handleDynamicInputTensor(ir::IOIndex{i}, desc);
  }

  assert(_output_tensors.size() == desc.outputs.size());
  for (uint32_t i = 0; i < _output_tensors.size(); ++i)
  {
    // TODO Remove dynamic_cast
    auto tensor = std::dynamic_pointer_cast<backend::controlflow::UserTensor>(_output_tensors[i]);
    assert(tensor);
    tensor->set_dynamic(); // It can't be resized but shape could change
    // TODO Better design for ITensor? (we need const_cast as ITensor is writable)
    tensor->setBuffer(static_cast<uint8_t *>(const_cast<void *>(desc.outputs[i]->buffer)),
                      desc.outputs[i]->size);
  }

  executeImpl();

  // Update output(s) desc
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

    auto dyn_tensor_manager = _input_tensors[io_ind.value()]->dynamic_tensor_manager();
    assert(dyn_tensor_manager);
    dyn_tensor_manager->applyShape(operand_ind, changed_input_shape);
  }
}

bool ExecutorBase::hasDynamicInput()
{
  for (auto &tensor : _input_tensors)
  {
    if (tensor->is_dynamic())
      return true;
  }
  return false;
}

} // namespace exec
} // namespace onert
