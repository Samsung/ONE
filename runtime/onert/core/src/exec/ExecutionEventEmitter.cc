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

#include "ExecutionEventEmitter.h"

namespace onert
{
namespace exec
{

void ExecutionEventEmitter::addListener(std::unique_ptr<IExecutionEventListener> listener)
{
  _listeners.emplace_back(std::move(listener));
}

void ExecutionEventEmitter::notifyExecuteOpEnd(ir::SubgraphIndex s, ir::OperationIndex o,
                                               ir::OpCode opcode, backend::ITensor *tensor)
{
  for (auto &l : _listeners)
  {
    l->handleExecuteOpEnd(s, o, opcode, tensor);
  }
}

void ExecutionEventEmitter::notifyExecuteSubgEnd(ir::SubgraphIndex s)
{
  for (auto &l : _listeners)
  {
    l->handleExecuteSubgEnd(s);
  }
}

} // namespace exec
} // namespace onert
