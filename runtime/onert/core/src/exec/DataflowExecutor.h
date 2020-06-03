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

#ifndef __ONERT_EXEC_DATAFLOW_EXECUTOR_H__
#define __ONERT_EXEC_DATAFLOW_EXECUTOR_H__

#include <list>
#include <map>
#include <unordered_map>

#include "exec/FunctionSequence.h"
#include "Job.h"
#include "ir/OperandIndexSequence.h"
#include "ir/Index.h"
#include <memory>
#include "exec/ExecutorBase.h"
#include "compiler/CodeMap.h"

namespace onert
{
namespace exec
{

class DataflowExecutor : public ExecutorBase
{

protected:
  virtual void notify(uint32_t finished_job_id);
  bool noWaitingJobs();

public:
  /**
   * @brief Constructs a DataflowExecutor object
   *
   * @param lowered_graph LoweredGraph object
   * @param tensor_builders Tensor builders that are currently used
   * @param code_map OpSequence and its code map
   */
  DataflowExecutor(std::unique_ptr<ir::LoweredGraph> lowered_graph,
                   const backend::TensorBuilderSet &tensor_builders, compiler::CodeMap &&code_map);

  void executeImpl() override;

protected:
  int64_t calculateRank(const std::vector<ir::OperationIndex> &operations);
  void emplaceToReadyJobs(const uint32_t &id);

protected:
  compiler::CodeMap _code_map;
  /**
   * @brief A vector of finished jobs for current execution
   *        After a run it has all the jobs of this execution for the next run
   */
  std::vector<std::unique_ptr<Job>> _finished_jobs;
  /**
   * @brief A vector of waiting jobs for current execution
   *        All the jobs are moved from #_finished_jobs to it when start a run
   */
  std::vector<std::unique_ptr<Job>> _waiting_jobs;
  /**
   * @brief Jobs' output info
   *        Used for notifying after finishing a job
   */
  std::vector<std::list<uint32_t>> _output_info;
  std::vector<uint32_t> _initial_input_info;
  std::vector<uint32_t> _input_info;
  /**
   * @brief A collection of jobs that are ready for execution
   *        Jobs in it are ready to be scheduled.
   *        Ordered by priority from `_indexed_ranks`
   */
  std::multimap<int64_t, std::unique_ptr<Job>, std::greater<int64_t>> _ready_jobs;

  /// @brief Which job runs which op and function.
  std::unordered_map<uint32_t, ir::OpSequenceIndex> _job_to_op_seq;
};

} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_DATAFLOW_EXECUTOR_H__
