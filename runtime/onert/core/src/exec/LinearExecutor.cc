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
    auto profiling_subg_index = _tracing_ctx->getSubgraphIndex(&_graph);

    _subject.notifySubgraphBegin(profiling_subg_index);
    for (auto &&code : _code)
    {
      const auto backend = code.lower_info->backend();
// TODO : Move ruy profiler into ExecutionObserver
#ifdef RUY_PROFILER
      ruy::profiler::ScopeLabel label(code.op->name());
#endif
      _subject.notifyJobBegin(this, profiling_subg_index, code.op_ind, backend);

      auto &fn_seq = code.fn_seq;

      fn_seq->initRunning();

      bool handle_dynamic_tensor =
        _lowered_graph->isDynamicTensor(code.op_ind) || hasDynamicInput();
      fn_seq->enableDynamicShapeInferer(handle_dynamic_tensor);
      fn_seq->run();

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

      auto &fn_seq = code.fn_seq;

      fn_seq->initRunning();

      bool handle_dynamic_tensor =
        _lowered_graph->isDynamicTensor(code.op_ind) || hasDynamicInput();
      fn_seq->enableDynamicShapeInferer(handle_dynamic_tensor);
      fn_seq->run();
    }
  }
}

} // namespace exec
} // namespace onert
