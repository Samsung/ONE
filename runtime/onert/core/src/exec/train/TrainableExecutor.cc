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
  const compiler::train::TensorRegistries &tensor_regs,
  compiler::train::TrainableCodeMap &&code_map,
  const std::vector<ir::OperationIndex> &forward_order,
  const std::vector<ir::OperationIndex> &backward_order, const util::TracingCtx *tracing_ctx,
  const ir::train::LossInfo &loss_info)
  : _code_map{std::move(code_map)}, _forward_order{std::move(forward_order)},
    _backward_order{std::move(backward_order)}, _lowered_graph{std::move(lowered_graph)},
    _backend_contexts{std::move(backend_contexts)},
    _trainable_graph{_lowered_graph->trainable_graph()}, _tensor_regs{std::move(tensor_regs)},
    _mutex(), _tracing_ctx(tracing_ctx), _loss_info(loss_info)
{
  auto build_tensor_list = [&](const auto &ind_seq, auto &tensors) {
    assert(tensors.empty());
    for (auto &&ind : ind_seq)
    {
      backend::ITensor *tensor = _tensor_regs.getITensor(ind);
      assert(tensor != nullptr);
      auto io_tensor = nnfw::misc::polymorphic_downcast<backend::builtin::IOTensor *>(tensor);
      tensors.push_back(io_tensor);
    }
  };
  build_tensor_list(_trainable_graph.getInputs(), _input_tensors);
  build_tensor_list(_trainable_graph.getOutputs(), _output_tensors);
}

void TrainableExecutor::execute(const std::vector<backend::IPortableTensor *> &,
                                const std::vector<backend::IPortableTensor *> &)
{
  throw std::runtime_error("TrainableExecutor does not support multiple subgraphs yet");
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

  if (!training)
  {
    // Set output(s)
    assert(_output_tensors.size() == desc.outputs.size());
    for (uint32_t i = 0; i < _output_tensors.size(); ++i)
    {
      auto tensor = _output_tensors[i];

      if (desc.outputs[i] == nullptr)
        throw std::runtime_error{"Output " + std::to_string(i) + "'s buffer is not set."};
      tensor->setUserTensor(static_cast<uint8_t *>(desc.outputs[i]->buffer), desc.outputs[i]->size);
    }
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
    for (auto &&index : _forward_order)
    {
      const auto &code = _code_map.at(index);
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
    for (auto &&index : _forward_order)
    {
      const auto &code = _code_map.at(index);
// TODO : Move ruy profiler into ExecutionObserver
#ifdef RUY_PROFILER
      ruy::profiler::ScopeLabel label(code.op->name());
#endif
      auto &tn_seq = code.tn_seq;
      tn_seq->forward(training);
    }
  }
}

void TrainableExecutor::backward(const IODescription &, uint32_t training_step)
{
  // For thread-safe, use mutex
  // TODO: if all used backends on this executor are thread-safe,
  //       do not need to use mutex (otherwise, use mutex)
  std::lock_guard<std::mutex> lock(_mutex);

  backwardImpl(training_step);
}

void TrainableExecutor::backwardImpl(uint32_t training_step)
{
  if (_tracing_ctx)
  {
    auto profiling_subg_index = _tracing_ctx->getSubgraphIndex(&_trainable_graph.graph());

    _subject.notifySubgraphBegin(profiling_subg_index);
    for (auto &&index : _backward_order)
    {
      const auto &code = _code_map.at(index);
      const auto backend = code.lower_info->backend();
// TODO : Move ruy profiler into ExecutionObserver
#ifdef RUY_PROFILER
      ruy::profiler::ScopeLabel label(code.op->name());
#endif
      _subject.notifyJobBegin(this, profiling_subg_index, code.op_ind, backend);

      auto &tn_seq = code.tn_seq;
      tn_seq->backward(training_step);

      _subject.notifyJobEnd(this, profiling_subg_index, code.op_ind, backend);
    }
    _subject.notifySubgraphEnd(profiling_subg_index);
  }
  else
  {
    for (auto &&index : _backward_order)
    {
      const auto &code = _code_map.at(index);
// TODO : Move ruy profiler into ExecutionObserver
#ifdef RUY_PROFILER
      ruy::profiler::ScopeLabel label(code.op->name());
#endif
      auto &tn_seq = code.tn_seq;
      tn_seq->backward(training_step);
    }
  }
}

float TrainableExecutor::getLoss(const ir::IOIndex &pred_io_ind) const
{
  const auto &loss_ind = _trainable_graph.getLossIndex(pred_io_ind);
  if (loss_ind.undefined())
    throw std::runtime_error{"Loss " + std::to_string(loss_ind.value()) + " is not defined."};
  backend::ITensor *tensor = _tensor_regs.getITensor(loss_ind);
  long double sum = 0;
  for (uint64_t i = 0; i < tensor->getShape().num_elements(); ++i)
  {
    sum += reinterpret_cast<float *>(tensor->buffer())[i];
  }
  if (_loss_info.reduction_type == ir::train::LossReductionType::SumOverBatchSize)
  {
    sum /= tensor->getShape().num_elements();
  }
  return static_cast<float>(sum);
}

void TrainableExecutor::iterateTrainableTensors(
  const std::function<void(const ir::OperandIndex &, const backend::train::ITrainableTensor *)> &fn)
  const
{
  _tensor_regs.iterateTrainableTensors(fn);
}

} // namespace train
} // namespace exec
} // namespace onert
