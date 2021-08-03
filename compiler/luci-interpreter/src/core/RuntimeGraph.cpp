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

#include "core/RuntimeGraph.h"

#include "core/RuntimeModule.h"

#include <algorithm>
#include <unordered_map>

namespace luci_interpreter
{

class RuntimeGraph::TensorAllocPlan
{
  std::vector<std::vector<Tensor *>> _alloc_plan;
  std::vector<std::vector<Tensor *>> _dealloc_plan;
  bool _valid = false;
  MManager *_memory_manager;

public:
  explicit TensorAllocPlan(MManager *memory_manager);
  void invalidate() { _valid = false; }
  bool isValid() const { return _valid; }
  void build(const RuntimeGraph &graph);
  void allocate(size_t kernel_index) const;
  void deallocate(size_t kernel_index) const;
};

RuntimeGraph::TensorAllocPlan::TensorAllocPlan(MManager *memory_manager)
  : _memory_manager(memory_manager)
{
}

void RuntimeGraph::TensorAllocPlan::build(const RuntimeGraph &graph)
{
  invalidate();
  using Lifetime = std::pair<size_t, size_t>;
  std::unordered_map<Tensor *, Lifetime> lifetimes;
  const size_t num_kernels = graph._kernels.size();
  for (size_t index = 0; index < num_kernels; ++index)
  {
    const auto &kernel = graph._kernels[index];
    for (const Tensor *tensor : kernel->getInputTensors())
    {
      auto nc_tensor = const_cast<Tensor *>(tensor);
      if (lifetimes.count(nc_tensor) > 0)
        lifetimes.at(nc_tensor).second = index;
    }
    for (Tensor *tensor : kernel->getOutputTensors())
    {
      assert(lifetimes.count(tensor) == 0);
      lifetimes[tensor] = Lifetime(index, index);
    }
  }
  for (const Tensor *tensor : graph.getOutputTensors())
  {
    auto nc_tensor = const_cast<Tensor *>(tensor);
    if (lifetimes.count(nc_tensor) > 0)
      lifetimes.at(nc_tensor).second = num_kernels;
  }
  _alloc_plan.assign(num_kernels, std::vector<Tensor *>());
  _dealloc_plan.assign(num_kernels + 1, std::vector<Tensor *>());
  for (const auto &item : lifetimes)
  {
    _alloc_plan[item.second.first].push_back(item.first);
    _dealloc_plan[item.second.second].push_back(item.first);
  }
  _valid = true;
}

void RuntimeGraph::TensorAllocPlan::allocate(size_t kernel_index) const
{
  assert(_valid && kernel_index < _alloc_plan.size());
  for (Tensor *tensor : _alloc_plan[kernel_index])
  {
    _memory_manager->allocate_memory(tensor);
  }
}

void RuntimeGraph::TensorAllocPlan::deallocate(size_t kernel_index) const
{
  assert(_valid && kernel_index < _dealloc_plan.size());
  for (Tensor *tensor : _dealloc_plan[kernel_index])
  {
    _memory_manager->release_memory(tensor);
  }
}

RuntimeGraph::RuntimeGraph(RuntimeModule *owning_module, MManager *memory_manager)
  : _owning_module(owning_module),
    _tensor_alloc_plan(std::make_unique<TensorAllocPlan>(memory_manager))
{
}

RuntimeGraph::~RuntimeGraph() {}

Tensor *RuntimeGraph::addTensor(std::unique_ptr<Tensor> &&tensor)
{
  assert(tensor != nullptr);
  _tensors.push_back(std::move(tensor));
  return _tensors.back().get();
}

void RuntimeGraph::setInputTensors(const std::vector<Tensor *> &input_tensors)
{
  assert(std::all_of(input_tensors.cbegin(), input_tensors.cend(),
                     [](Tensor *tensor) { return tensor != nullptr; }));
  _input_tensors = input_tensors;
}

void RuntimeGraph::setOutputTensors(const std::vector<Tensor *> &output_tensors)
{
  assert(std::all_of(output_tensors.cbegin(), output_tensors.cend(),
                     [](Tensor *tensor) { return tensor != nullptr; }));
  _output_tensors = output_tensors;
}

void RuntimeGraph::addKernel(std::unique_ptr<Kernel> &&kernel)
{
  assert(kernel != nullptr);
  _kernels.push_back(std::move(kernel));
  _tensor_alloc_plan->invalidate();
}

void RuntimeGraph::execute() const
{
  if (!_tensor_alloc_plan->isValid())
    _tensor_alloc_plan->build(*this);

  EventNotifier *event_notifier = _owning_module->getEventNotifier();

  // Notify the observers that the input tensors have changed.
  if (event_notifier != nullptr)
  {
    for (const Tensor *input_tensor : getInputTensors())
    {
      event_notifier->postTensorWrite(input_tensor);
    }
  }

  for (size_t index = 0; index < _kernels.size(); ++index)
  {
    const auto &kernel = _kernels[index];
    if (event_notifier != nullptr)
    {
      event_notifier->preOperatorExecute(kernel.get());
    }

    // TODO The `configure` method should only be called if the outputs of an operator need to be
    //  resized.
    kernel->configure();

    // Preallocate outputs in advance instead of relying on automatic allocation
    _tensor_alloc_plan->allocate(index);

    kernel->execute();

    if (event_notifier != nullptr)
    {
      event_notifier->postOperatorExecute(kernel.get());
    }

    for (const Tensor *tensor : kernel->getOutputTensors())
    {
      if (event_notifier != nullptr)
      {
        event_notifier->postTensorWrite(tensor);
      }
    }
    _tensor_alloc_plan->deallocate(index);
  }
}

} // namespace luci_interpreter
