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

#include "LinearExecutor.h"
#ifdef RUY_PROFILER
#include "ruy/profiler/instrumentation.h"
#endif

namespace onert
{
namespace exec
{

void LinearExecutor::executeImpl()
{
  if (_tracing_ctx)
  {
    auto subg_idx = _tracing_ctx->getSubgraphIndex(&_graph);

    _subject.notifySubgraphBegin(subg_idx);
    for (auto &&code : _code)
    {
      const auto backend = code.lower_info->backend();
// TODO : Move ruy profiler into ExecutionObserver
#ifdef RUY_PROFILER
      ruy::profiler::ScopeLabel label(code.op->name());
#endif
      _subject.notifyJobBegin(this, subg_idx, code.op_ind, backend);

      auto &fn_seq = code.fn_seq;

      fn_seq->initRunning();

      bool handle_dynamic_tensor =
        _lowered_graph->getHasDynamicTensor(code.op_ind) || hasDynamicInput();
      fn_seq->enableDynamicShapeInferer(handle_dynamic_tensor);
      fn_seq->run();

      _subject.notifyJobEnd(this, subg_idx, code.op_ind, backend);

      auto op_idx = code.op_ind;
      auto &backend_ctx = _backend_contexts.at(backend);
      const auto &tensor_reg = backend_ctx->tensor_registry;
      const auto &op = _graph.operations().at(op_idx);
      const auto &outputs = op.getOutputs();
      const int kMaxOutTensors = 8;
      backend::ITensor *out_tensors[kMaxOutTensors];
      if (outputs.size() > kMaxOutTensors)
        throw std::runtime_error("Up to " + std::to_string(kMaxOutTensors) +
                                 " output are supported for execution listener");
      for (uint32_t i = 0; i < outputs.size(); ++i)
        out_tensors[i] = tensor_reg->getITensor(outputs.at(i));
      auto opcode = code.op->opcode();
      notifyExecuteOpEnd(subg_idx, op_idx, opcode, out_tensors, outputs.size());
    }
    _subject.notifySubgraphEnd(subg_idx);
    // NOTE: notifyExecuteSubgEnd is introduced to dump gathered minmax map.
    // For single model and single subgraph, it works.
    // To support, multiple subgraph (with if and while layer), it would be better
    // to dump at the end of model execution. Then, this code should be moved to
    // ExecutorBase::execute().
    notifyExecuteSubgEnd(subg_idx);
  }
  else
  {
    for (auto &&code : _code)
    {
// TODO : Move ruy profiler into ExecutionObserver
#ifdef RUY_PROFILER
      ruy::profiler::ScopeLabel label(code.op->name());
#endif

      auto &fn_seq = code.fn_seq;

      fn_seq->initRunning();

      bool handle_dynamic_tensor =
        _lowered_graph->getHasDynamicTensor(code.op_ind) || hasDynamicInput();
      fn_seq->enableDynamicShapeInferer(handle_dynamic_tensor);
      fn_seq->run();
    }
  }
}

} // namespace exec
} // namespace onert
