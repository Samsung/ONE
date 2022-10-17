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

#ifndef __ONERT_EXEC_EXECUTION_EVENT_EMITTER__
#define __ONERT_EXEC_EXECUTION_EVENT_EMITTER__

#include "IExecutionEventEmitter.h"
#include "IExecutionEventListener.h"

#include <list>
#include <memory>

namespace onert
{
namespace exec
{

/* ExecutionEventEmitter (hereafter, EEE) is different from ExecutionObserver (hereafter EO) in:
 *
 * 1. EEE is used in release code, not only for debug
 * 2. EEE is interested in the output tensor value. It has different paramters.
 * 3. EEE will consider multiple model, subgraph later.
 */
class ExecutionEventEmitter : public IExecutionEventEmitter
{
public:
  void addListener(std::unique_ptr<IExecutionEventListener> listener) override;
  void notifyExecuteOpEnd(ir::SubgraphIndex, ir::OperationIndex, ir::OpCode,
                          backend::ITensor *) override;
  void notifyExecuteSubgEnd(ir::SubgraphIndex) override;

private:
  std::list<std::unique_ptr<IExecutionEventListener>> _listeners;
};

} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_EXECUTION_EVENT_EMITTER__
