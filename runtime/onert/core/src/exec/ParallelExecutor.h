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

#ifndef __ONERT_EXEC_PARALLEL_EXECUTOR_H__
#define __ONERT_EXEC_PARALLEL_EXECUTOR_H__

#include "DataflowExecutor.h"
#include "ParallelScheduler.h"

#include "util/TracingCtx.h"

#include <memory>

namespace onert
{
namespace exec
{

/**
 * @brief Class to execute Graph in parallel
 */
class ParallelExecutor : public DataflowExecutor
{
protected:
  void notify(uint32_t finished_job_id) override;

public:
  /**
   * @brief Constructs a ParallelExecutor object
   *
   * @param lowered_graph LoweredGraph object
   * @param tensor_builders Tensor builders that are currently used
   * @param code_map @c ir::Operation and its code map
   */
  ParallelExecutor(std::unique_ptr<compiler::LoweredGraph> lowered_graph,
                   backend::BackendContexts &&backend_contexts,
                   const compiler::TensorRegistries &tensor_regs, compiler::CodeMap &&code_map,
                   const util::TracingCtx *tracing_ctx);

  void executeImpl(const ExecutionObservee &subject) override;

private:
  std::condition_variable _cv_jobs;
  std::mutex _mu_jobs;
  std::unique_ptr<ParallelScheduler> _scheduler;
};

} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_PARALLEL_EXECUTOR_H__
