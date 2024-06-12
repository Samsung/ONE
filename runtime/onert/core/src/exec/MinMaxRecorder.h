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

#ifndef __ONERT_EXEC_MINMAX_RECORDER__
#define __ONERT_EXEC_MINMAX_RECORDER__

#include "ExecutionObservers.h"
#include "ir/Index.h"
#include "exec/MinMaxMap.h"

#include <string>

namespace onert
{
namespace exec
{

class MinMaxRecorder : public IExecutionObserver
{
public:
  MinMaxRecorder(const std::string &workspace_dir, const ir::Graph &graph,
                 const backend::BackendContexts &backend_contexts);
  void handleJobBegin(IExecutor *, ir::SubgraphIndex, ir::OperationIndex,
                      const backend::Backend *) override
  {
    return;
  }
  void handleJobEnd(IExecutor *, ir::SubgraphIndex, ir::OperationIndex,
                    const backend::Backend *) override;
  void handleSubgraphBegin(ir::SubgraphIndex) override;
  void handleSubgraphEnd(ir::SubgraphIndex) override;

private:
  const ir::Graph &_graph;
  const backend::BackendContexts &_backend_contexts;
  std::string _workspace_dir;
  OpMinMaxMap _op_minmax;
  IOMinMaxMap _input_minmax;
};

} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_MINMAX_RECORDER__
