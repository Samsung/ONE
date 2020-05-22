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

#include "backend/controlflow/operand/Tensor.h"

namespace onert
{
namespace exec
{

ExecutorBase::ExecutorBase(std::unique_ptr<ir::LoweredGraph> &&lowered_graph,
                           const std::vector<std::shared_ptr<backend::ITensor>> &input_tensors,
                           const std::vector<std::shared_ptr<backend::ITensor>> &output_tensors,
                           const backend::TensorBuilderSet &tensor_builders)
    : _lowered_graph{std::move(lowered_graph)}, _graph{_lowered_graph->graph()},
      _input_tensors{input_tensors}, _output_tensors{output_tensors}, _mutex()
{
  // TODO Fix the way of knowing whether it is primary or not
  bool primary_executor = !(_input_tensors.empty() && _output_tensors.empty());
  if (!primary_executor)
  {
    auto build_tensor_list = [&](const onert::ir::OperandIndexSequence &ind_seq) {
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

    _input_tensors = build_tensor_list(_graph.getInputs());
    _output_tensors = build_tensor_list(_graph.getOutputs());
  }

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

void ExecutorBase::changeInputShape(const ir::OperandIndex &index, const ir::Shape &new_shape)
{
  for (auto &input_tensor : _input_tensors)
  {
    auto dyn_alloc_info = _input_to_dyn_alloc_info.find(input_tensor);
    if (dyn_alloc_info == _input_to_dyn_alloc_info.end())
      continue;

    // when user-provided input change is stored in _input_to_dyn_alloc_info
    if (index == dyn_alloc_info->second.ind)
    {
      auto dyn_tensor_manager = dyn_alloc_info->second.dyn_tensor_manager;
      assert(dyn_tensor_manager);
      dyn_tensor_manager->changeShape(index, new_shape);
      return;
    }
  }
  throw std::runtime_error("changeInputShape(): Cannot find such index or "
                           "check if the tensor's backend supports dynamic tensor.");
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

  assert(_input_tensors.size() == desc.inputs.size());
  for (uint32_t i = 0; i < _input_tensors.size(); ++i)
  {
    // TODO Remove dynamic_cast
    auto tensor =
        std::dynamic_pointer_cast<backend::controlflow::operand::UserTensor>(_input_tensors[i]);
    assert(tensor);
    // TODO Better design for ITensor? (we need const_cast as ITensor is writable)
    tensor->setBuffer(static_cast<uint8_t *>(const_cast<void *>(desc.inputs[i]->buffer)),
                      desc.inputs[i]->size);
  }

  assert(_output_tensors.size() == desc.outputs.size());
  for (uint32_t i = 0; i < _output_tensors.size(); ++i)
  {
    // TODO Remove dynamic_cast
    auto tensor =
        std::dynamic_pointer_cast<backend::controlflow::operand::UserTensor>(_output_tensors[i]);
    assert(tensor);
    // TODO Better design for ITensor? (we need const_cast as ITensor is writable)
    tensor->setBuffer(static_cast<uint8_t *>(const_cast<void *>(desc.outputs[i]->buffer)),
                      desc.outputs[i]->size);
  }

  executeImpl();
}

} // namespace exec
} // namespace onert
