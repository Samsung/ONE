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

#ifndef __ONERT_EXEC_EXECUTION_OBSERVEE_H__
#define __ONERT_EXEC_EXECUTION_OBSERVEE_H__

#include <list>

#include "exec/ExecutionObservers.h"
#include "ir/Index.h"

namespace onert
{
namespace exec
{

/**
 * @brief Class that
 *
 */
class ExecutionObservee
{
public:
  /**
   * @brief Register an observer
   *
   * @param observer Observer to be added
   */
  void add(std::unique_ptr<IExecutionObserver> observer);
  void notifySubgraphBegin(ir::SubgraphIndex ind);
  void notifySubgraphEnd(ir::SubgraphIndex ind);
  void notifyJobBegin(IExecutor *executor, ir::SubgraphIndex subg_ind, ir::OperationIndex op_ind,
                      const backend::Backend *backend);
  void notifyJobEnd(IExecutor *executor, ir::SubgraphIndex subg_ind, ir::OperationIndex op_ind,
                    const backend::Backend *backend);

private:
  std::list<std::unique_ptr<IExecutionObserver>> _observers;
};

} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_EXECUTION_OBSERVEE__
