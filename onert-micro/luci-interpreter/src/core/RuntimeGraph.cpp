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
#include "luci_interpreter/memory_managers/StaticMemoryManager.h"

#include <algorithm>
#include <map>

namespace luci_interpreter
{

// BaseRuntimeGraph
BaseRuntimeGraph::BaseRuntimeGraph(IMemoryManager *memory_manager) : _memory_manager(memory_manager)
{
}

Tensor *BaseRuntimeGraph::addTensor(std::unique_ptr<Tensor> &&tensor)
{
  assert(tensor != nullptr);
  _tensors.push_back(std::move(tensor));
  return _tensors.back().get();
}

AffineQuantization *
BaseRuntimeGraph::addAffineQuantization(std::unique_ptr<AffineQuantization> &&quantization)
{
  assert(quantization != nullptr);
  _affine_quantizations.push_back(std::move(quantization));
  return _affine_quantizations.back().get();
}

void BaseRuntimeGraph::addInputTensor(Tensor *input_tensor)
{
  _input_tensors.push_back(input_tensor);
}

void BaseRuntimeGraph::addOutputTensor(Tensor *output_tensor)
{
  _output_tensors.push_back(output_tensor);
}

void BaseRuntimeGraph::configureAllocations(Tensor *tensor)
{
  _memory_manager->allocate_memory(*tensor);
}

void BaseRuntimeGraph::addKernel(std::unique_ptr<Kernel> &&kernel)
{
  assert(kernel != nullptr);
  _kernels.push_back(std::move(kernel));
  _is_valid = false;
}

// RuntimeGraph
RuntimeGraph::RuntimeGraph(IMemoryManager *memory_manager) : BaseRuntimeGraph(memory_manager) {}

RuntimeGraph::~RuntimeGraph()
{
  for (auto &tensor : _tensors)
  {
    if (tensor->is_data_allocated())
      _memory_manager->release_memory(*tensor);
  }
}

void RuntimeGraph::buildAllocDeallocPlan()
{
  invalidate();
  using Lifetime = std::pair<size_t, size_t>;
  std::map<Tensor *, Lifetime> lifetimes;
  const size_t num_kernels = _kernels.size();
  for (size_t index = 0; index < num_kernels; ++index)
  {
    const auto &kernel = _kernels[index];
    for (const Tensor *tensor : kernel->getInputTensors())
    {
      auto nc_tensor = const_cast<Tensor *>(tensor);
      if (lifetimes.count(nc_tensor) > 0)
      {
        if (kernel->getInplaceValue())
          lifetimes.at(nc_tensor).second = -1;
        else
          lifetimes.at(nc_tensor).second = index;
      }
    }
    for (Tensor *tensor : kernel->getOutputTensors())
    {
      assert(lifetimes.count(tensor) == 0);
      if (kernel->getInplaceValue())
        lifetimes[tensor] = Lifetime(-1, index);
      else
        lifetimes[tensor] = Lifetime(index, index);
    }
  }
  for (const Tensor *tensor : getOutputTensors())
  {
    auto nc_tensor = const_cast<Tensor *>(tensor);
    if (lifetimes.count(nc_tensor) > 0)
      lifetimes.at(nc_tensor).second = num_kernels;
  }
  _alloc_plan.assign(num_kernels, std::vector<Tensor *>());
  _dealloc_plan.assign(num_kernels + 1, std::vector<Tensor *>());
  for (const auto &item : lifetimes)
  {
    if (item.second.first != -1)
      _alloc_plan[item.second.first].push_back(item.first);
    if (item.second.second != -1)
      _dealloc_plan[item.second.second].push_back(item.first);
  }
  _is_valid = true;
}

void RuntimeGraph::allocate(size_t kernel_index) const
{
  assert(_is_valid && kernel_index < _alloc_plan.size());
  for (Tensor *tensor : _alloc_plan[kernel_index])
  {
    _memory_manager->allocate_memory(*tensor);
  }
}

void RuntimeGraph::deallocate(size_t kernel_index) const
{
  assert(_is_valid && kernel_index < _dealloc_plan.size());
  for (Tensor *tensor : _dealloc_plan[kernel_index])
  {
    _memory_manager->release_memory(*tensor);
  }
}

void RuntimeGraph::configure()
{
  if (not _is_valid)
    buildAllocDeallocPlan();

  for (auto &kernel : _kernels)
  {
    kernel->configure();
  }

  _is_valid = true;
}

void RuntimeGraph::execute()
{
  if (not _is_valid)
    configure();

  for (size_t index = 0; index < _kernels.size(); ++index)
  {
    const auto &kernel = _kernels[index];

    // TODO: add kernel->configure for methods with dynamic shapes

    // Preallocate outputs in advance instead of relying on automatic allocation
    allocate(index);

    kernel->execute();

    deallocate(index);
  }
}

// StaticRuntimeGraph
StaticRuntimeGraph::StaticRuntimeGraph(IMemoryManager *memory_manager)
  : BaseRuntimeGraph(memory_manager)
{
}

StaticRuntimeGraph::~StaticRuntimeGraph()
{
  // Release intermediate computing buffer.
  const auto static_memory_manager = dynamic_cast<StaticMemoryManager *>(_memory_manager);
  if (static_memory_manager->is_owning_buffers())
  {
    static_memory_manager->release_computing_buf();
    static_memory_manager->release_input_buf();
    static_memory_manager->release_output_buf();
  }
}

void StaticRuntimeGraph::configure()
{
  // Allocate memory for intermediate computing buffer and for output buffer.
  const auto static_memory_manager = dynamic_cast<StaticMemoryManager *>(_memory_manager);
  if (static_memory_manager->is_owning_buffers())
  {
    static_memory_manager->allocate_computing_buf();
    static_memory_manager->allocate_output_buf();
  }

  // Set tensor's data pointer for intermediate tensors
  for (auto &kernel : _kernels)
  {
    const auto output_tensors = kernel->getOutputTensors();

    for (auto tensor : output_tensors)
      _memory_manager->allocate_memory(*tensor);
  }

  // Set tensor's data pointer for output tensors
  if (static_memory_manager->is_owning_buffers())
  {
    for (const auto output_tensor : _output_tensors)
    {
      static_memory_manager->allocate_memory_for_output(*output_tensor);
    }
  }

  _is_valid = true;
}

void StaticRuntimeGraph::configure_kernels()
{
  for (auto &kernel : _kernels)
  {
    kernel->configure();
  }
}

void StaticRuntimeGraph::execute()
{
  if (not _is_valid)
    configure();

  for (auto &kernel : _kernels)
  {
    // TODO: add kernel->configure for methods with dynamic shapes
    kernel->execute();
  }

  // Release intermediate computing buffer.
  const auto static_memory_manager = dynamic_cast<StaticMemoryManager *>(_memory_manager);
  if (static_memory_manager->is_owning_buffers())
    static_memory_manager->release_computing_buf();

  _is_valid = false;
}

} // namespace luci_interpreter
