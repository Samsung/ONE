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

void ExecutionObservee::add(std::unique_ptr<IExecutionObserver> observer)
{
  _observers.emplace_back(std::move(observer));
}

void ExecutionObservee::notifySubgraphBegin(ir::SubgraphIndex ind)
{
  for (auto &o : _observers)
  {
    o->handleSubgraphBegin(ind);
  }
}

void ExecutionObservee::notifySubgraphEnd(ir::SubgraphIndex ind)
{
  for (auto &o : _observers)
  {
    o->handleSubgraphEnd(ind);
  }
}

void ExecutionObservee::notifyJobBegin(IExecutor *executor, ir::SubgraphIndex index,
                                       const ir::OpSequence *op_seq,
                                       const backend::Backend *backend)
{
  for (auto &o : _observers)
  {
    o->handleJobBegin(executor, index, op_seq, backend);
  }
}

void ExecutionObservee::notifyJobEnd(IExecutor *executor, ir::SubgraphIndex index,
                                     const ir::OpSequence *op_seq, const backend::Backend *backend)
{
  for (auto &o : _observers)
  {
    o->handleJobEnd(executor, index, op_seq, backend);
  }
}

} // namespace exec
} // namespace onert
