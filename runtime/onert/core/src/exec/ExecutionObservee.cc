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

#include "ExecutionObservee.h"

namespace onert
{
namespace exec
{

ExecutionObservee::ExecutionObservee(const ExecObservers &observers)
{
  // TODO Use execution option
  if (util::getConfigBool(util::config::MINMAX_DUMP))
  {
    auto observer = observers.get(ObserverType::MINMAX_DUMP);
    if (!observer)
      throw std::runtime_error{"MinMaxRecorder is only supported on LinearExecutor, single model"};

    _observers.emplace_back(observer);
  }

  if (util::getConfigBool(util::config::TRACING_MODE))
  {
    auto observer = observers.get(ObserverType::TRACING);
    if (!observer)
      throw std::runtime_error{"Cannot find TracingObserver"};

    _observers.emplace_back(observer);
  }

  if (util::getConfigBool(util::config::PROFILING_MODE))
  {
    auto observer = observers.get(ObserverType::PROFILE);
    if (!observer)
      throw std::runtime_error{
        "Profiling is only supported on DataflowExecutor with heterogenous scheduler"};

    _observers.emplace_back(observer);
  }
}

void ExecutionObservee::notifySubgraphBegin(ir::SubgraphIndex ind) const
{
  for (auto &&o : _observers)
  {
    o->handleSubgraphBegin(ind);
  }
}

void ExecutionObservee::notifySubgraphEnd(ir::SubgraphIndex ind) const
{
  for (auto &&o : _observers)
  {
    o->handleSubgraphEnd(ind);
  }
}

void ExecutionObservee::notifyJobBegin(IExecutor *executor, ir::SubgraphIndex subg_ind,
                                       ir::OperationIndex op_ind,
                                       const backend::Backend *backend) const
{
  for (auto &&o : _observers)
  {
    o->handleJobBegin(executor, subg_ind, op_ind, backend);
  }
}

void ExecutionObservee::notifyJobEnd(IExecutor *executor, ir::SubgraphIndex subg_ind,
                                     ir::OperationIndex op_ind,
                                     const backend::Backend *backend) const
{
  for (auto &&o : _observers)
  {
    o->handleJobEnd(executor, subg_ind, op_ind, backend);
  }
}

} // namespace exec
} // namespace onert
