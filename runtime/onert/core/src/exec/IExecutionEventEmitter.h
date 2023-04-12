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

#ifndef __ONERT_EXEC_IEXECUTION_EVENT_EMITTER__
#define __ONERT_EXEC_IEXECUTION_EVENT_EMITTER__

#include "ir/Index.h"
#include "ir/OpCode.h"

#include <memory>

namespace onert
{
namespace backend
{
class ITensor;
}
} // namespace onert

namespace onert
{
namespace exec
{
struct IExecutionEventListener;
struct IExecutionEventEmitter
{
  /**
   * @brief Add a listener
   *
   * @param listener Listener to be added
   */
  virtual void addListener(std::unique_ptr<IExecutionEventListener> listener) = 0;
  virtual void notifyExecuteOpEnd(ir::SubgraphIndex, ir::OperationIndex, ir::OpCode,
                                  backend::ITensor **, uint32_t) = 0;
  virtual void notifyExecuteSubgEnd(ir::SubgraphIndex) = 0;
  virtual ~IExecutionEventEmitter() = default;
};
} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_IEXECUTION_EVENT_EMITTER__
