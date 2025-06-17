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

#include "util/ConfigSource.h"
#include <algorithm>
#include <misc/polymorphic_downcast.h>

namespace onert::exec
{

ExecutorBase::ExecutorBase(std::unique_ptr<compiler::LoweredGraph> &&lowered_graph,
                           backend::BackendContexts &&backend_contexts,
                           const compiler::TensorRegistries &tensor_regs,
                           const util::TracingCtx *tracing_ctx)
  : _lowered_graph{std::move(lowered_graph)}, _backend_contexts{std::move(backend_contexts)},
    _graph{_lowered_graph->graph()}, _mutex(), _tracing_ctx(tracing_ctx)
{
  auto build_tensor_list = [&](const auto &ind_seq, auto &tensors) {
    assert(tensors.empty());
    for (auto &&ind : ind_seq)
    {
      backend::builtin::IOTensor *io_tensor = tensor_regs.getIOTensor(ind);
      assert(io_tensor != nullptr);
      tensors.push_back(io_tensor);
    }
  };
  build_tensor_list(_graph.getInputs(), _input_tensors);
  build_tensor_list(_graph.getOutputs(), _output_tensors);
}

void ExecutorBase::execute(const std::vector<backend::IPortableTensor *> &inputs,
                           const std::vector<backend::IPortableTensor *> &outputs,
                           const ExecutionOptions &options)
{
  // For thread-safe, use mutex
  // TODO: if all used backends on this executor are thread-safe,
  //       do not need to use mutex (otherwise, use mutex)
  // Deadlock occurs when an Executor is called recursively.
  std::lock_guard<std::mutex> lock(_mutex);
  _current_options = options;

  assert(inputs.size() == _graph.getInputs().size());
  assert(inputs.size() == _input_tensors.size());
  for (uint32_t n = 0; n < inputs.size(); ++n)
  {
    const auto input = inputs[n];
    assert(input->buffer() != nullptr || input->get_info().total_size() == 0);
    auto input_tensor = _input_tensors[n];
    assert(input_tensor != nullptr);
    input_tensor->setTensor(input);
  }

  assert(outputs.size() == _graph.getOutputs().size());
  assert(outputs.size() == _output_tensors.size());
  for (uint32_t n = 0; n < outputs.size(); ++n)
  {
    const auto output = outputs[n];
    auto output_tensor = _output_tensors[n];
    assert(output_tensor != nullptr);
    assert(output->buffer() != nullptr || output->get_info().total_size() == 0 ||
           output_tensor->hasBackendTensor());
    if (!output_tensor->hasBackendTensor())
      output_tensor->setTensor(output);
  }

  // Create observee
  ExecutionObservee subject(_observers, options);

  executeImpl(subject);

  for (uint32_t n = 0; n < outputs.size(); ++n)
  {
    auto output_tensor = _output_tensors[n];
    assert(output_tensor != nullptr);
    if (output_tensor->hasBackendTensor())
    {
      assert(output_tensor->buffer() != nullptr);
      output_tensor->syncInfoFromBackendTensor();
    }
  }
}

bool ExecutorBase::hasDynamicInput()
{
  for (auto &&tensor : _input_tensors)
  {
    if (tensor->is_dynamic())
      return true;
  }
  return false;
}

} // namespace onert::exec
