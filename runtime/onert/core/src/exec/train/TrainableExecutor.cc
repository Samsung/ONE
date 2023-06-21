/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "TrainableExecutor.h"
#ifdef RUY_PROFILER
#include "ruy/profiler/instrumentation.h"
#endif

#include <misc/polymorphic_downcast.h>

namespace onert
{
namespace exec
{
namespace train
{

TrainableExecutor::TrainableExecutor(
  std::unique_ptr<compiler::train::LoweredTrainableGraph> lowered_graph,
  backend::train::TrainableBackendContexts &&backend_contexts,
  const compiler::TensorRegistries &tensor_regs, compiler::train::CodeMap &&code_map,
  const std::vector<ir::OperationIndex> &order, const util::TracingCtx *tracing_ctx)
  : _lowered_graph{std::move(lowered_graph)}, _backend_contexts{std::move(backend_contexts)},
    _trainable_graph{_lowered_graph->trainable_graph()}, _mutex(), _tracing_ctx(tracing_ctx)
{
  auto build_tensor_list = [&](const auto &ind_seq, auto &tensors) {
    assert(tensors.empty());
    for (auto ind : ind_seq)
    {
      backend::ITensor *tensor = tensor_regs.getITensor(ind);
      assert(tensor != nullptr);
      auto io_tensor = nnfw::misc::polymorphic_downcast<backend::builtin::IOTensor *>(tensor);
      tensors.push_back(io_tensor);
    }
  };
  build_tensor_list(_trainable_graph.getInputs(), _input_tensors);
  build_tensor_list(_trainable_graph.getOutputs(), _output_tensors);

  for (auto index : order)
  {
    auto &trainable_code = code_map.at(index);
    _code.emplace_back(std::move(trainable_code));
  }
}

void TrainableExecutor::execute(const std::vector<backend::IPortableTensor *> &,
                                const std::vector<backend::IPortableTensor *> &)
{
  throw std::runtime_error("TrainableExecutor does not support multiple subgraphs yet");
}

void TrainableExecutor::execute(const IODescription &desc)
{
  forward(desc, false);
}

void TrainableExecutor::forward(const IODescription &desc, bool training)
{
  // For thread-safe, use mutex
  // TODO: if all used backends on this executor are thread-safe,
  //       do not need to use mutex (otherwise, use mutex)
  std::lock_guard<std::mutex> lock(_mutex);

  // TODO Update IO tensors if desc has dynamic input
  // Set input(s)
  assert(_input_tensors.size() == desc.inputs.size());
  for (uint32_t i = 0; i < _input_tensors.size(); ++i)
  {
    auto tensor = _input_tensors[i];

    // TODO Check if (desc.inputs[i] == nullptr)
    // TODO Better design for ITensor? (we need const_cast as ITensor is writable)
    tensor->setUserTensor(static_cast<uint8_t *>(const_cast<void *>(desc.inputs[i]->buffer)),
                          desc.inputs[i]->size);
  }

  // Set output(s)
  assert(_output_tensors.size() == desc.outputs.size());
  for (uint32_t i = 0; i < _output_tensors.size(); ++i)
  {
    auto tensor = _output_tensors[i];

    if (desc.outputs[i] == nullptr)
      throw std::runtime_error{"Output " + std::to_string(i) + "'s buffer is not set."};
    tensor->setUserTensor(static_cast<uint8_t *>(desc.outputs[i]->buffer), desc.outputs[i]->size);
  }

  forwardImpl(training);

  // TODO Update output(s) desc if desc has dynamic input
}

void TrainableExecutor::forwardImpl(bool training)
{
  if (_tracing_ctx)
  {
    auto profiling_subg_index = _tracing_ctx->getSubgraphIndex(&_trainable_graph.graph());

    _subject.notifySubgraphBegin(profiling_subg_index);
    for (auto &&code : _code)
    {
      const auto backend = code.lower_info->backend();
// TODO : Move ruy profiler into ExecutionObserver
#ifdef RUY_PROFILER
      ruy::profiler::ScopeLabel label(code.op->name());
#endif
      _subject.notifyJobBegin(this, profiling_subg_index, code.op_ind, backend);

      auto &tn_seq = code.tn_seq;

      tn_seq->forward(training);

      _subject.notifyJobEnd(this, profiling_subg_index, code.op_ind, backend);
    }
    _subject.notifySubgraphEnd(profiling_subg_index);
  }
  else
  {
    for (auto &&code : _code)
    {
// TODO : Move ruy profiler into ExecutionObserver
#ifdef RUY_PROFILER
      ruy::profiler::ScopeLabel label(code.op->name());
#endif

      auto &tn_seq = code.tn_seq;

      tn_seq->forward(training);
    }
  }
}

void TrainableExecutor::backward(const IODescription &desc)
{
  // For thread-safe, use mutex
  // TODO: if all used backends on this executor are thread-safe,
  //       do not need to use mutex (otherwise, use mutex)
  std::lock_guard<std::mutex> lock(_mutex);

  // TODO Update IO tensors if desc has dynamic input
  // Set input(s)
  assert(_input_tensors.size() == desc.inputs.size());
  for (uint32_t i = 0; i < _input_tensors.size(); ++i)
  {
    auto tensor = _input_tensors[i];

    // TODO Check if (desc.inputs[i] == nullptr)
    // TODO Better design for ITensor? (we need const_cast as ITensor is writable)
    tensor->setUserTensor(static_cast<uint8_t *>(const_cast<void *>(desc.inputs[i]->buffer)),
                          desc.inputs[i]->size);
  }

  // Set output(s)
  assert(_output_tensors.size() == desc.outputs.size());
  for (uint32_t i = 0; i < _output_tensors.size(); ++i)
  {
    auto tensor = _output_tensors[i];

    if (desc.outputs[i] == nullptr)
      throw std::runtime_error{"Output " + std::to_string(i) + "'s buffer is not set."};
    tensor->setUserTensor(static_cast<uint8_t *>(desc.outputs[i]->buffer), desc.outputs[i]->size);
  }

  backwardImpl();

  // TODO Update output(s) desc if desc has dynamic input

  // TODO Support backwarding
}

void TrainableExecutor::backwardImpl()
{
  if (_tracing_ctx)
  {
    auto profiling_subg_index = _tracing_ctx->getSubgraphIndex(&_trainable_graph.graph());

    _subject.notifySubgraphBegin(profiling_subg_index);
    for (auto it = _code.rbegin(); it != _code.rend(); ++it)
    {
      const auto &code = *it;
      const auto backend = it->lower_info->backend();
// TODO : Move ruy profiler into ExecutionObserver
#ifdef RUY_PROFILER
      ruy::profiler::ScopeLabel label(code.op->name());
#endif
      _subject.notifyJobBegin(this, profiling_subg_index, code.op_ind, backend);

      auto &tn_seq = code.tn_seq;

      tn_seq->backward();

      _subject.notifyJobEnd(this, profiling_subg_index, code.op_ind, backend);
    }
    _subject.notifySubgraphEnd(profiling_subg_index);
  }
  else
  {
    for (auto it = _code.rbegin(); it != _code.rend(); ++it)
    {
      const auto &code = *it;
// TODO : Move ruy profiler into ExecutionObserver
#ifdef RUY_PROFILER
      ruy::profiler::ScopeLabel label(code.op->name());
#endif
      UNUSED_RELEASE(code);
      // TODO Support forwarding

      auto &tn_seq = code.tn_seq;

      tn_seq->backward();
    }
  }
}

} // namespace train
} // namespace exec
} // namespace onert
