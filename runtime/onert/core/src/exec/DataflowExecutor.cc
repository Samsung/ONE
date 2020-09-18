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
  auto &op_seq = _lowered_graph->op_seqs().at(_job_to_op_seq[job->index()]);
  auto rank = calculateRank(op_seq.operations());
  _ready_jobs.emplace(rank, std::move(job));
}

void DataflowExecutor::notify(uint32_t finished_job_id)
{
  for (auto id : _output_info[finished_job_id])
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
                                   const compiler::TensorRegistries &tensor_regs,
                                   compiler::CodeMap &&code_map)
    : ExecutorBase{std::move(lowered_graph), tensor_regs}, _code_map{std::move(code_map)}
{
  VERBOSE(DataflowExecutor) << "Constructing Dataflow Executor" << std::endl;

  const auto &op_seqs = _lowered_graph->op_seqs();
  // Assign jobs convert OpSequenceIndex to job index(uint32_t)
  uint32_t next_job_index = 0;
  std::unordered_map<ir::OpSequenceIndex, uint32_t> op_seq_to_job;
  op_seqs.iterate([&](const ir::OpSequenceIndex &op_seq_index, const ir::OpSequence &) {
    VERBOSE(DataflowExecutor) << "Create a job #" << next_job_index << " with OpSequenceIndex "
                              << op_seq_index.value() << std::endl;
    _finished_jobs.emplace_back(
        std::make_unique<Job>(next_job_index, _code_map.at(op_seq_index).fn_seq.get()));
    op_seq_to_job[op_seq_index] = next_job_index++;
  });

  _waiting_jobs.resize(next_job_index);
  _output_info.resize(next_job_index);
  _initial_input_info.resize(next_job_index, 0);

  op_seqs.iterate([&](const ir::OpSequenceIndex &op_seq_index, const ir::OpSequence &op_seq) {
    auto job_index = op_seq_to_job[op_seq_index];
    for (auto output : op_seq.getOutputs())
    {
      // Update output and input info
      op_seqs.iterate(
          [&](const ir::OpSequenceIndex &op_seq_cur_index, const ir::OpSequence &op_seq_cur) {
            if (op_seq_cur.getInputs().contains(output))
            {
              auto dep_index = op_seq_to_job[op_seq_cur_index];
              ++_initial_input_info[dep_index];
              _output_info[job_index].push_back(dep_index);
            }
          });
    }
  });
  for (const auto &s : op_seq_to_job)
    _job_to_op_seq.emplace(s.second, s.first);

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

  _subject.notifyModelBegin(this);

  while (!_ready_jobs.empty())
  {
    auto job = std::move((_ready_jobs.begin())->second);
    _ready_jobs.erase(_ready_jobs.begin());
    auto job_index = job->index();
    VERBOSE(DataflowExecutor) << "Run job #" << job_index << std::endl;

    auto op_seq_index = _job_to_op_seq[job_index];
    auto op_seq = &_lowered_graph->op_seqs().at(op_seq_index);
    const backend::Backend *backend =
        _lowered_graph->getLowerInfo()->op_seq.at(op_seq_index)->backend();

    _subject.notifyJobBegin(this, op_seq, backend);

    job->fn_seq()->initRunning();

    // check if FunctionSequence needs to handle dynamic tensor
    bool handle_dynamic_tensor = op_seq->has_dynamic_tensor() || dynamic_input_exists;
    job->fn_seq()->enableDynamicShapeInferer(handle_dynamic_tensor);

    job->run();

    _subject.notifyJobEnd(this, op_seq, backend);
    notify(job_index);
    _finished_jobs[job_index] = std::move(job);
  }
  assert(noWaitingJobs());

  _subject.notifyModelEnd(this);

  // Reset input info for the next execution
  _input_info = _initial_input_info;
}

} // namespace exec
} // namespace onert
