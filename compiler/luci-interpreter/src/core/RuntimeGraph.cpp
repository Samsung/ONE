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
}

void RuntimeGraph::configure()
{
  // Configure the kernels, e.g. resize the tensors that they produce and do other kernel dependent
  // initialization. This has to be done in execution order, because configuration of a kernel may
  // (and in most cases does) depend on configurations of its predecessors.
  // TODO Some kernels (ex. Reshape, Pad) need some of their input tensors (ex 'shape', 'paddings')
  //  to be known in order to configure properly. This means that 'configure'  and 'execute' steps
  //  should be interleaved. For now such 'dynamic' tensors are not supported.
  for (const auto &kernel : _kernels)
  {
    kernel->configure();
  }
}

void RuntimeGraph::execute() const
{
  EventNotifier *event_notifier = _owning_module->getEventNotifier();

  // Notify the observers that the input tensors have changed.
  if (event_notifier != nullptr)
  {
    for (const Tensor *input_tensor : getInputTensors())
    {
      event_notifier->postTensorWrite(input_tensor);
    }
  }

  for (const auto &kernel : _kernels)
  {
    if (event_notifier != nullptr)
    {
      event_notifier->preOperatorExecute(kernel.get());
    }

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
  }
}

} // namespace luci_interpreter
