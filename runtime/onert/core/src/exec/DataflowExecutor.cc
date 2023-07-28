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

#include "DataflowExecutor.h"

#include <cassert>

#include "util/logging.h"

namespace onert
{
namespace exec
{

int64_t DataflowExecutor::calculateRank(const std::vector<ir::OperationIndex> &operations)
{
  int64_t rank = 0;
  if (!_indexed_ranks)
  {
    return rank;
  }
  for (const auto &operation_idx : operations)
  {
    auto it = _indexed_ranks->find(operation_idx);
    if (it == _indexed_ranks->end())
    {
      assert(_graph.operations().at(operation_idx).opcode() == ir::OpCode::Permute &&
             operations.size() == 1);
      // run Permute ASAP for next operations to be ready for other backends
      return std::numeric_limits<int64_t>::max();
    }
    else
    {
      rank += it->second;
    }
  }
  return rank;
}

void DataflowExecutor::emplaceToReadyJobs(const uint32_t &id)
{
  auto &job = _waiting_jobs[id];
  assert(job != nullptr);
  auto rank = calculateRank({_job_to_op[job->index()]});
  _ready_jobs.emplace(rank, std::move(job));
}

void DataflowExecutor::notify(uint32_t finished_job_id)
{
  for (auto &&id : _output_info[finished_job_id])
  {
    assert(_input_info[id] > 0);
    auto count = --_input_info[id];
    if (count == 0) // No dependent jobs left, ready for execution
    {
      emplaceToReadyJobs(id);
    }
  }
}
bool DataflowExecutor::noWaitingJobs()
{
  return std::all_of(_waiting_jobs.begin(), _waiting_jobs.end(),
                     [](const std::unique_ptr<Job> &job) { return job == nullptr; });
}

DataflowExecutor::DataflowExecutor(std::unique_ptr<compiler::LoweredGraph> lowered_graph,
                                   backend::BackendContexts &&backend_contexts,
                                   const compiler::TensorRegistries &tensor_regs,
                                   compiler::CodeMap &&code_map,
                                   const util::TracingCtx *tracing_ctx)
  : ExecutorBase{std::move(lowered_graph), std::move(backend_contexts), tensor_regs, tracing_ctx},
    _code_map{std::move(code_map)}
{
  VERBOSE(DataflowExecutor) << "Constructing Dataflow Executor" << std::endl;

  // Assign jobs convert OperationIndex to job index(uint32_t)
  uint32_t next_job_index = 0;
  std::unordered_map<ir::OperationIndex, uint32_t> op_to_job;
  const auto &operations = _lowered_graph->graph().operations();
  operations.iterate([&](const ir::OperationIndex &op_ind, const ir::IOperation &) {
    VERBOSE(DataflowExecutor) << "Create a job " << next_job_index << " with Operation " << op_ind
                              << std::endl;
    _finished_jobs.emplace_back(
      std::make_unique<Job>(next_job_index, _code_map.at(op_ind).fn_seq.get()));
    op_to_job[op_ind] = next_job_index++;
  });

  _waiting_jobs.resize(next_job_index);
  _output_info.resize(next_job_index);
  _initial_input_info.resize(next_job_index, 0);

  operations.iterate([&](const ir::OperationIndex &op_ind, const ir::IOperation &op) {
    auto job_index = op_to_job[op_ind];
    for (auto &&output : op.getOutputs())
    {
      // Update output and input info
      operations.iterate([&](const ir::OperationIndex &op_cur_ind, const ir::IOperation &op_cur) {
        if (op_cur.getInputs().contains(output))
        {
          auto dep_index = op_to_job[op_cur_ind];
          ++_initial_input_info[dep_index];
          _output_info[job_index].push_back(dep_index);
        }
      });
    }
  });
  for (const auto &s : op_to_job)
    _job_to_op.emplace(s.second, s.first);

  _input_info = _initial_input_info;
}

void DataflowExecutor::executeImpl()
{
  assert(noWaitingJobs());

  bool dynamic_input_exists = hasDynamicInput();

  // Execution setup
  _waiting_jobs.swap(_finished_jobs); // Move finished jobs to waiting jobs

  for (uint32_t i = 0; i < _waiting_jobs.size(); ++i)
  {
    if (_input_info[i] == 0)
    {
      emplaceToReadyJobs(i);
    }
  }
  assert(!_ready_jobs.empty()); // Cannot begin if there is no initial jobs

  auto profiling_subg_index = _tracing_ctx->getSubgraphIndex(&_graph);

  _subject.notifySubgraphBegin(profiling_subg_index);

  while (!_ready_jobs.empty())
  {
    auto job = std::move((_ready_jobs.begin())->second);
    _ready_jobs.erase(_ready_jobs.begin());
    auto job_index = job->index();
    VERBOSE(DataflowExecutor) << "Run job " << job_index << std::endl;

    auto op_ind = _job_to_op[job_index];
    const backend::Backend *backend = _lowered_graph->lower_info().operation.at(op_ind).backend();

    _subject.notifyJobBegin(this, profiling_subg_index, op_ind, backend);

    job->fn_seq()->initRunning();

    // check if FunctionSequence needs to handle dynamic tensor
    bool handle_dynamic_tensor =
      _lowered_graph->getHasDynamicTensor(op_ind) || dynamic_input_exists;
    job->fn_seq()->enableDynamicShapeInferer(handle_dynamic_tensor);

    job->run();

    _subject.notifyJobEnd(this, profiling_subg_index, op_ind, backend);
    notify(job_index);
    _finished_jobs[job_index] = std::move(job);
  }
  assert(noWaitingJobs());

  _subject.notifySubgraphEnd(profiling_subg_index);

  // Reset input info for the next execution
  _input_info = _initial_input_info;
}

} // namespace exec
} // namespace onert
