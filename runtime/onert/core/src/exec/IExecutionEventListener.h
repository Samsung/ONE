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

#ifndef __ONERT_EXEC_IEXECUTION_EVENT_LISTENER__
#define __ONERT_EXEC_IEXECUTION_EVENT_LISTENER__

#include "ir/Index.h"
#include "ir/OpCode.h"

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

struct IExecutionEventListener
{
  virtual void handleExecuteOpEnd(ir::SubgraphIndex, ir::OperationIndex, ir::OpCode,
                                  backend::ITensor *) = 0;
  virtual void handleExecuteSubgEnd(ir::SubgraphIndex) = 0;
};

} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_IEXECUTION_EVENT_LISTENER__
