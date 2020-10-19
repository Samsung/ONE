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
  _tensor_dealloc_plan.clear();
}

void RuntimeGraph::prepareDeallocPlan() const
{
  // For every tensor which is an output of some kernel find the last kernel which uses it.
  std::unordered_map<const Tensor *, size_t> dealloc_index;
  for (size_t index = 0; index < _kernels.size(); ++index)
  {
    const auto &kernel = _kernels[index];
    for (const Tensor *tensor : kernel->getOutputTensors())
    {
      assert(dealloc_index.count(tensor) == 0);
      dealloc_index[tensor] = index;
    }
    for (const Tensor *tensor : kernel->getInputTensors())
      if (dealloc_index.count(tensor) > 0)
        dealloc_index[tensor] = index;
  }
  for (const Tensor *tensor : getOutputTensors())
    dealloc_index.erase(tensor);
  _tensor_dealloc_plan.assign(_kernels.size(), std::vector<Tensor *>());
  for (const auto &item : dealloc_index)
    _tensor_dealloc_plan[item.second].push_back(const_cast<Tensor *>(item.first));
}

void RuntimeGraph::execute() const
{
  if (_tensor_dealloc_plan.empty())
    prepareDeallocPlan();

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

    // Deallocate tensors which are not to be used any more in this run
    for (Tensor *tensor : _tensor_dealloc_plan[index])
      tensor->deallocate();
  }
}

} // namespace luci_interpreter
